#include <iostream>
#include <utility>
#include <string>
#include <fstream>

#include <libdash.h> // requires MPI, allows easy access to data containers

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define NBODY_CUDA 1
#else
#define NBODY_CUDA 0
#endif

#if BOOST_VERSION < 106700 && (__CUDACC__ || __IBMCPP__)
    #ifdef BOOST_PP_VARIADICS
        #undef BOOST_PP_VARIADICS
    #endif
    #define BOOST_PP_VARIADICS 1
#endif

#include <alpaka/alpaka.hpp> // used for abstract kernels, for GPUs and CPUs
#ifdef __CUDACC__
	#define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC __forceinline__
#else
	#define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC inline
#endif
#include <llama/llama.hpp> // for data structures, maps defined access pattern to one that is ideal for the machine
#include <random>

#include <AlpakaAllocator.hpp>
#include <AlpakaMemCopy.hpp>

#include "HRChrono.hpp"
#include "Dummy.hpp"
#include "Human.hpp"


/* ******************************************************************
 *               Define relevant constants here                     *
 ***************************************************************** */

#define NBODY_PROBLEM_SIZE 800
#define NBODY_BLOCK_SIZE 256
#define NBODY_STEPS 50000
#define DATA_DUMP_STEPS 100 // write data to file every N steps

using Element = float; // change to double if needed

constexpr Element EPS2 = 1e-10;

constexpr Element ts = 1e-8; // timestep in [s]

constexpr Element particleMass = 24*1.66053886E-27;// M(Mg)/A =  4.03594014⋅10−27 kg

constexpr Element particleCharge = 1.6021766209e-19; // [C]

constexpr Element voltage = 5.; // [V]

constexpr Element rmin = 0.; // position of potential minimum x=y=z=0

constexpr Element rNull = 1.5e-2; // [m]
constexpr Element rNullSquared = rNull*rNull; // [m]

constexpr Element phys_c        = 299792458.;
constexpr Element phys_emfactor = phys_c * phys_c * 1E-7;

/* *************************************************************** */


/*
HarmonicField:
// With a potential of                                                   //
//                                                                       //
// V(r) = ( U / r0^2 ) * (r-rmin)^2                                      //
//                                                                       //
// where U [V] is the voltage and r0 [m] is the potential depth          //
// the corresponding force on a particle of charge q [C] is              //
//                                                                       //
// hforce = q * ( - 2U / r0^2 ) * (r-rmin)                               //
//                                                                       //
// where - 2U / r0^2 is the restoring force factor for the harmonic      //
// potential V.                                                          //
//                                                                       //
U = 5V, r0 = 1µm, q = 2e = 3.204353e-19 C
[hforce] = [C*(V/m^2)*m] = [C*V/m] = [J/m] = [Nm/m] = [N]

CoulombForce, i<->j
F_i,j = _md_phys_emfactor * q^2 / ||pos(i)-pos(j)||_2^3 * (pos(i)-pos(j))

Equation of Motion
F_i = hforce_i + cforce_i
F = m * a => d^2x/dt^2 = F(t,x,dx/dt) / m
v += a * dt
pos += v * dt
*/

namespace dd
{
    struct Pos {};
    struct Vel {};
    struct X {};
    struct Y {};
    struct Z {};
    struct HForce {};
    struct CForce {};
//     struct Mass {};
}

struct particle
{
    struct
    {
        Element x, y, z;
    } pos;
    struct
    {
        Element x, y, z;
    } vel;
    struct
    {
        Element x, y, z;
    } cForce;
};

using Particle = llama::DS<
    llama::DE< dd::Pos, llama::DS< //position
        llama::DE< dd::X, Element >,
        llama::DE< dd::Y, Element >,
        llama::DE< dd::Z, Element >
    > >,
    llama::DE< dd::Vel,llama::DS< //velocity
        llama::DE< dd::X, Element >,
        llama::DE< dd::Y, Element >,
        llama::DE< dd::Z, Element >
    > >,
    llama::DE< dd::CForce,llama::DS< //coulomb forces
        llama::DE< dd::X, Element >,
        llama::DE< dd::Y, Element >,
        llama::DE< dd::Z, Element >
    > >
>;

template<
    typename T_VirtualDatum1,
    typename T_VirtualDatum2
>
LLAMA_FN_HOST_ACC_INLINE
auto
pPInteraction(
    T_VirtualDatum1&& localP, //self
    T_VirtualDatum2&& remoteP, //comparison
    Element const & ts
)
-> void
{
    // main computation for two elements
    Element const d[3] = {
        localP( dd::Pos(), dd::X() ) -
        remoteP( dd::Pos(), dd::X() ),
        localP( dd::Pos(), dd::Y() ) -
        remoteP( dd::Pos(), dd::Y() ),
        localP( dd::Pos(), dd::Z() ) -
        remoteP( dd::Pos(), dd::Z() )
    };

    Element distSqr = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    Element dist = sqrt(distSqr);
    Element distCube = distSqr * dist;

    Element forcefactor;

    // calculate coulomb force
    if ( distCube > 0. ){
        forcefactor = phys_emfactor * particleCharge *
            particleCharge / distCube;

        localP( dd::CForce(), dd::X() )  += forcefactor * d[0];
        localP( dd::CForce(), dd::Y() )  += forcefactor * d[1];
        localP( dd::CForce(), dd::Z() )  += forcefactor * d[2];
    }
}

template<
    typename T_VirtualDatum1
>
LLAMA_FN_HOST_ACC_INLINE
auto
cooling_linear(
    T_VirtualDatum1&& vk, //self
    Element const & vacc,
    Element const & vmax
)
-> Element
{
    Element restore = 1e-19; // [ C*V/m*s/m ]

    Element dv = vk - vmax;

    if ( vacc < 0. )
        return -1. * restore * dv;
    else if ( (dv < 0.) && (dv > -1. * vacc) )
        return +restore * (dv + vacc);
    else if ( (dv > 0.) && (dv < +vacc) )
        return -1. * restore * (dv - vacc);
    return 0.0;
}

template<
    typename T_Acc,
    std::size_t T_size,
    std::size_t T_counter,
    std::size_t threads
>
struct BlockSharedMemoryAllocator
{
    using type = common::allocator::AlpakaShared<
        T_Acc,
        T_size,
        T_counter
    >;

    template <
        typename T_Factory,
        typename T_Mapping
    >
    LLAMA_FN_HOST_ACC_INLINE
    static
    auto
    allocView(
        T_Mapping const mapping,
        T_Acc const & acc
    )
    -> decltype( T_Factory::allocView( mapping, acc ) )
    {
        return T_Factory::allocView( mapping, acc );
    };
};

template<
    typename T_Acc,
    std::size_t T_size,
    std::size_t T_counter
>
struct BlockSharedMemoryAllocator<
    T_Acc,
    T_size,
    T_counter,
    1
>
{
    using type = llama::allocator::Stack<
        T_size
    >;

    template <
        typename T_Factory,
        typename T_Mapping
    >
    LLAMA_FN_HOST_ACC_INLINE
    static
    auto
    allocView(
        T_Mapping const mapping,
        T_Acc const & acc
    )
    -> decltype( T_Factory::allocView( mapping ) )
    {
        return T_Factory::allocView( mapping );
    };
};

template<
    std::size_t problemSize,
    std::size_t elems,
    std::size_t blockSize
>

// called for a block of particles
struct ParticleInteractionKernel
{
    template<
        typename T_Acc,
        typename T_ViewLocal,
        typename T_ViewRemote
    >
    LLAMA_FN_HOST_ACC_INLINE
    void operator()(
        T_Acc const &acc,
        T_ViewLocal localParticles,
        T_ViewRemote remoteParticles,
        Element ts
    ) const
    {

        constexpr std::size_t threads = blockSize / elems;
        using SharedAllocator = BlockSharedMemoryAllocator<
            T_Acc,
            llama::SizeOf< typename decltype(remoteParticles)::Mapping::DatumDomain >::value
            * blockSize,
            __COUNTER__,
            threads
        >;


        using SharedMapping = llama::mapping::SoA<
            typename decltype(remoteParticles)::Mapping::UserDomain,
            typename decltype(remoteParticles)::Mapping::DatumDomain
        >;
        SharedMapping const sharedMapping( { blockSize } );

        using SharedFactory = llama::Factory<
            SharedMapping,
            typename SharedAllocator::type
        >;

        auto temp = SharedAllocator::template allocView<
            SharedFactory,
            SharedMapping
        >( sharedMapping, acc );

        auto threadIndex  = alpaka::idx::getIdx<
            alpaka::Grid,
            alpaka::Threads
        >( acc )[ 0u ];

        auto threadBlockIndex  = alpaka::idx::getIdx<
            alpaka::Block,
            alpaka::Threads
        >( acc )[ 0u ];

        auto const start = threadIndex * elems;
        auto const   end = alpaka::math::min(
            acc,
            start + elems,
            problemSize
        );
        LLAMA_INDEPENDENT_DATA
        for ( std::size_t b = 0; b < ( problemSize + blockSize -1 ) / blockSize; ++b )
        {
            auto const start2 = b * blockSize;
            auto const   end2 = alpaka::math::min(
                acc,
                start2 + blockSize,
                problemSize
            ) - start2;

            LLAMA_INDEPENDENT_DATA
            for (
                auto pos2 = decltype(end2)(0);
                pos2 + threadBlockIndex < end2;
                pos2 += threads
            )
                temp(pos2 + threadBlockIndex) = remoteParticles( start2 + pos2 + threadBlockIndex );

            // compute loop
            LLAMA_INDEPENDENT_DATA
            for ( auto pos2 = decltype(end2)(0); pos2 < end2; ++pos2 )
                LLAMA_INDEPENDENT_DATA
                for ( auto pos = start; pos < end; ++pos )
                    pPInteraction(
                        localParticles( pos ),
                        temp( pos2 ),
                        ts
                    );
        }
    }
};

template<
    std::size_t problemSize,
    std::size_t elems
>
struct SingleParticleKernel
{
    template<
        typename T_Acc,
        typename T_View
    >
    LLAMA_FN_HOST_ACC_INLINE
    void operator()(
        T_Acc const &acc,
        T_View particles,
        Element ts
    ) const
    {
        auto threadIndex  = alpaka::idx::getIdx<
            alpaka::Grid,
            alpaka::Threads
        >( acc )[ 0u ];

        auto const start = threadIndex * elems;
        auto const   end = alpaka::math::min(
            acc,
            (threadIndex + 1) * elems,
            problemSize
        );

        LLAMA_INDEPENDENT_DATA
        for ( auto pos = start; pos < end; ++pos )
        {

            // cooling laser force
            Element lForce[3] = {
                cooling_linear( particles( pos )( dd::Vel(), dd::X()),
                                -20.,
                                -10.) +
                cooling_linear( particles( pos )( dd::Vel(), dd::X()),
                                20.,
                                10.),
                cooling_linear( particles( pos )( dd::Vel(), dd::Y()),
                                -20.,
                                -10.) +
                cooling_linear( particles( pos )( dd::Vel(), dd::Y()),
                                20.,
                                10.),
                cooling_linear( particles( pos )( dd::Vel(), dd::Z()),
                                -20.,
                                -10.) +
                cooling_linear( particles( pos )( dd::Vel(), dd::Z()),
                                20.,
                                10.)
            };

            // harmonic forces
            Element hForce[3] = {
                particleCharge * ( -2.0 * voltage / rNullSquared) *
                    ( particles( pos )( dd::Pos(), dd::X()) - rmin ),
                particleCharge * ( -2.0 * voltage / rNullSquared) *
                    ( particles( pos )( dd::Pos(), dd::Y()) - rmin ),
                particleCharge * ( -2.0 * voltage / rNullSquared) *
                    ( particles( pos )( dd::Pos(), dd::Z()) - rmin )

            };


            // F_i = hforce_i + cforce_i + lforce_i
            Element const F_i[3] = {
                particles( pos )( dd::CForce(), dd::X() ) +
                    lForce[0] +
                    hForce[0],
                particles( pos )( dd::CForce(), dd::Y() ) +
                    lForce[1] +
                    hForce[1],
                particles( pos )( dd::CForce(), dd::Z() ) +
                    lForce[2] +
                    hForce[2]
            };

            // F = m * a => d^2x/dt^2 = F(t,x,dx/dt) / m
            Element const a[3] = {
                F_i[0] / particleMass,
                F_i[1] / particleMass,
                F_i[2] / particleMass,
            };

            // v += a * dt
            particles( pos )( dd::Vel(), dd::X() ) +=
                a[0] * ts; // ts = timestep
            particles( pos )( dd::Vel(), dd::Y() ) +=
                a[1] * ts;
            particles( pos )( dd::Vel(), dd::Z() ) +=
                a[2] * ts;

            // pos += v * dt
            particles( pos )( dd::Pos(), dd::X() ) +=
                particles( pos )( dd::Vel(), dd::X() ) * ts; // ts = timestep
            particles( pos )( dd::Pos(), dd::Y() ) +=
                particles( pos )( dd::Vel(), dd::Y() ) * ts;
            particles( pos )( dd::Pos(), dd::Z() ) +=
                particles( pos )( dd::Vel(), dd::Z() ) * ts;

            // reset coulomb forces
            particles( pos )( dd::CForce(), dd::X() )  = 0;
            particles( pos )( dd::CForce(), dd::Y() )  = 0;
            particles( pos )( dd::CForce(), dd::Z() )  = 0;


        }
    }
};

template<
    typename T_Acc,
    std::size_t blockSize,
    std::size_t hardwareThreads
>
struct ThreadsElemsDistribution
{
    static constexpr std::size_t elemCount = blockSize;
    static constexpr std::size_t threadCount = 1u;
};

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<
        std::size_t blockSize,
        std::size_t hardwareThreads,
        typename T_Dim,
        typename T_Size
    >
    struct ThreadsElemsDistribution<
        alpaka::acc::AccGpuCudaRt<T_Dim, T_Size>,
        blockSize,
        hardwareThreads
    >
    {
        static constexpr std::size_t elemCount = 1u;
        static constexpr std::size_t threadCount = blockSize;
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    template<
        std::size_t blockSize,
        std::size_t hardwareThreads,
        typename T_Dim,
        typename T_Size
    >
    struct ThreadsElemsDistribution<
        alpaka::acc::AccCpuOmp2Threads<T_Dim, T_Size>,
        blockSize,
        hardwareThreads
    >
    {
        static constexpr std::size_t elemCount =
            ( blockSize + hardwareThreads - 1u ) / hardwareThreads;
        static constexpr std::size_t threadCount = hardwareThreads;
    };
#endif

template<
    typename T_Parameter
>
struct PassThroughAllocator
{
    using PrimType = unsigned char;
    using BlobType = PrimType*;
    using Parameter = T_Parameter*;

    LLAMA_NO_HOST_ACC_WARNING
    static inline
    auto
    allocate(
        std::size_t count,
        Parameter const pointer
    )
    -> BlobType
    {
        return reinterpret_cast<BlobType>(pointer);
    }
};


int main(int argc,char * * argv)
{
    // ALPAKA
    using Dim = alpaka::dim::DimInt< 1 >;
    using Size = std::size_t;
    using Extents = Size;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;

#if NBODY_CUDA == 1
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
#else
    //~ using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
    //~ using Acc = alpaka::acc::AccCpuOmp2Threads<Dim, Size>;
    //~ using Acc = alpaka::acc::AccCpuOmp4<Dim, Size>;
#endif // NBODY_CUDA
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
#if NBODY_CUDA == 1
    using Queue = alpaka::queue::QueueCudaRtSync;
#else
    using Queue = alpaka::queue::QueueCpuSync;
#endif // NBODY_CUDA
    DevAcc const devAcc( alpaka::pltf::getDevByIdx< PltfAcc >( 0u ) );
    DevHost const devHost( alpaka::pltf::getDevByIdx< PltfHost >( 0u ) );
    Queue queue( devAcc ) ;

    // abstraction to distribute computation, 1D array (?)
    dash::init(&argc, &argv);

    dart_unit_t myid = dash::myid();
    dart_unit_t size = dash::size();

    // NBODY
    constexpr std::size_t problemSize = NBODY_PROBLEM_SIZE;
    constexpr std::size_t blockSize = NBODY_BLOCK_SIZE;
    constexpr std::size_t hardwareThreads = 2; //relevant for OpenMP2Threads
    using Distribution = ThreadsElemsDistribution<
        Acc,
        blockSize,
        hardwareThreads
    >;
    constexpr std::size_t elemCount = Distribution::elemCount;
    constexpr std::size_t threadCount = Distribution::threadCount;
    constexpr std::size_t steps = NBODY_STEPS;

    //DASH
    dash::Array<particle> particles;

    // LLAMA
    using UserDomain = llama::UserDomain< 1 >;
    const UserDomain userDomainSize{ problemSize };

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Particle
    >;
    Mapping const mapping( userDomainSize );

    using DevFactory = llama::Factory<
        Mapping,
        common::allocator::Alpaka<
            DevAcc,
            Size
        >
    >;
    using MirrorFactory = llama::Factory<
        Mapping,
        common::allocator::AlpakaMirror<
            DevAcc,
            Size,
            Mapping
        >
    >;
    using HostFactory = llama::Factory<
        Mapping,
        common::allocator::Alpaka<
            DevHost,
            Size
        >
    >;
    using LocalFactory = llama::Factory<
        Mapping,
        PassThroughAllocator<
            particle
        >
    >;

    if (myid == 0) {
        std::cout << (size * problemSize) / 1000 << " thousand particles (";
        std::cout << human_readable(size * (problemSize * llama::SizeOf<Particle>::value)) << ")\n";
    }

    HRChrono chrono;

    particles.allocate(size * problemSize);
    auto   hostView = LocalFactory::allocView( mapping, particles.lbegin() );
    auto    devView =    DevFactory::allocView( mapping,  devAcc );
    auto mirrowView = MirrorFactory::allocView( mapping, devView );

    // will be used as double buffer for remote->host and host->device copying
    auto   remoteHostView =   HostFactory::allocView( mapping, devHost );
    auto    remoteDevView =    DevFactory::allocView( mapping,  devAcc );
    auto remoteMirrowView = MirrorFactory::allocView( mapping, devView );

    chrono.printAndReset("Alloc:");

    std::default_random_engine generator;
    std::normal_distribution< Element > distribution(
        Element( 0 ), // mean
        Element( 5e-4 )  // stddev
    );
    // TODO: set 1 to sigma from TMDUtils.cpp
    // TODO: initialize vel with that sigma
    auto seed = distribution(generator);
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < problemSize; ++i)
    {
        // initialize position in X, Y, Z with random
        seed = distribution(generator);
        hostView(i)(dd::Pos(), dd::X()) = seed;

        seed = distribution(generator);
        hostView(i)(dd::Pos(), dd::Y()) = seed;

        seed = distribution(generator);
        hostView(i)(dd::Pos(), dd::Z()) = seed;

        // initialize velocity in X, Y, Z with zero (approximation)
        hostView(i)(dd::Vel(), dd::X()) = 0;
        hostView(i)(dd::Vel(), dd::Y()) = 0;
        hostView(i)(dd::Vel(), dd::Z()) = 0;

        // initialize coulomb force in X, Y, Z with zero
        hostView(i)(dd::CForce(), dd::X()) = 0;
        hostView(i)(dd::CForce(), dd::Y()) = 0;
        hostView(i)(dd::CForce(), dd::Z()) = 0;

        // initialize mass with constant
//         hostView(i)(dd::Mass()) = particleMass;
        /*
        std::cout << hostView(i)(dd::Pos(), dd::X()) \
            << "\t" \
            << hostView(i)(dd::Pos(), dd::Y()) \
            << "\t" \
            << hostView(i)(dd::Pos(), dd::Z()) \
            << "\t" \
            << hostView(i)(dd::Vel(), dd::X()) \
            << "\t" \
            << hostView(i)(dd::HForce(), dd::X()) \
            << "\t" \
            << hostView(i)(dd::Mass()) \
            << std::endl;
        */
    }

    chrono.printAndReset("Init:");

    alpaka::mem::view::ViewPlainPtr<DevHost, unsigned char, Dim, Size> hostPlain(
        reinterpret_cast<unsigned char*>(particles.lbegin()), devHost, problemSize * llama::SizeOf<Particle>::value);
    alpaka::mem::view::copy(queue,
        devView.blob[0].buffer,
        hostPlain,
        problemSize * llama::SizeOf<Particle>::value);

    chrono.printAndReset("Copy H->D");

    const alpaka::vec::Vec< Dim, Size > elems (
        static_cast< Size >( elemCount )
    );
    const alpaka::vec::Vec< Dim, Size > threads (
        static_cast< Size >( threadCount )
    );
    constexpr auto innerCount = elemCount * threadCount;
    const alpaka::vec::Vec< Dim, Size > blocks (
        static_cast< Size >( ( problemSize + innerCount - 1 ) / innerCount )
    );

    // calculate workload distribution
    auto const workdiv = alpaka::workdiv::WorkDivMembers<
        Dim,
        Size
    > {
        blocks,
        threads,
        elems
    };

    // copy hostView to devView
    ParticleInteractionKernel<
        problemSize,
        elemCount,
        blockSize
    > particleInteractionKernel;
    SingleParticleKernel<
        problemSize,
        elemCount
    > singleParticleKernel;

    // initialize progress bar
    Element progress;
    if ( myid == 0 ){
        std::cout << "Progress:\n";
        std::cout << "0%.....................50%...................100%\n";
        progress = 0.02;
    }

    for ( std::size_t s = 0; s < steps; ++s)
    {
        //std::cout<<(Element)s/(Element)steps<<std::endl;
        if ( ((Element)s/(Element)steps)>=progress && myid == 0 ){
            progress+=0.02;
            std::cout << "."<<std::flush;
        }

        // dump data to file, print first and last step and in given interval

        if ( s == 0 || s % DATA_DUMP_STEPS == 0 || s == (steps - 1) ){
            if ( myid == 0 ){
                std::string i = std::to_string(s);
                std::string fileName = "data";
                fileName.append(i);
                fileName.append(".csv");
                std::fstream myfile;
                myfile.open(fileName, std::fstream::out | std::fstream::trunc);

                for (std::size_t i = 0; i < problemSize; ++i)
                {
                    myfile << hostView(i)(dd::Pos(), dd::X()) \
                        << "," \
                        << hostView(i)(dd::Pos(), dd::Y()) \
                        << "," \
                        << hostView(i)(dd::Pos(), dd::Z()) \
                        << std::endl;
                }
                myfile.close();
            }
        }

        /* pair-wise with local particles */
        alpaka::kernel::exec< Acc > (
            queue,
            workdiv,
            particleInteractionKernel,
            mirrowView,
            mirrowView,
            ts
        );

        //chrono.printAndReset("Update kernel:       ");

        /* pair-wise with remote particles */
        for (dart_unit_t unit_it = 1; unit_it < size; ++unit_it)
        {
            dart_unit_t remote = (myid + unit_it) % size;

            // get remote local block into remoteHostView
            auto remote_begin = particles.begin() + (remote * problemSize);
            auto remote_end   = remote_begin + problemSize;
            auto target_begin = reinterpret_cast<particle*>(alpaka::mem::view::getPtrNative(remoteHostView.blob[0].buffer));
            dash::copy(remote_begin, remote_end, target_begin); // copy particles from remote

            //chrono.printAndReset("Copy from remote:    ");

            alpakaMemCopy( remoteDevView, remoteHostView, userDomainSize, queue );

            alpaka::kernel::exec< Acc > (
                queue,
                workdiv,
                particleInteractionKernel,
                mirrowView,
                remoteMirrowView,
                ts
            );

            //chrono.printAndReset("Update remote kernel:");

        }

        // move kernel
        alpaka::kernel::exec<Acc>(
            queue,
            workdiv,
            singleParticleKernel,
            mirrowView,
            ts
        );
        //chrono.printAndReset("Move kernel:         ");

        dummy( static_cast<void*>( mirrowView.blob[0] ) );

            alpaka::mem::view::copy(queue,
                hostPlain,
                devView.blob[0].buffer,
                problemSize * llama::SizeOf<Particle>::value);

            particles.barrier();

    }

    std::cout<<std::endl; // at the end of progress bar

    chrono.printAndReset("Copy D->H");

    dash::finalize();

    return 0;
}
