#include <iostream>
#include <utility>
#include <string>
#include <fstream>
#include <math.h>

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
#include "allreduce.h"


/* ******************************************************************
 *               Define relevant constants here                     *
 ***************************************************************** */

#define FACTOR 3

//#define NBODY_PROBLEM_SIZE 2048
#define NBODY_PROBLEM_SIZE 500
#define NBODY_BLOCK_SIZE 128
#define NBODY_STEPS 200000 * FACTOR
#define DATA_DUMP_STEPS 100 * FACTOR // write data to file every N steps
#define RESIDUUM 0.0001 // finish simulation at this residuum (max. velocity in [m/s])

using Element = float; // change to double if needed

constexpr Element EPS2 = 1e-10;

constexpr Element ts = 1e-8 / FACTOR; // timestep in [s]

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
    struct CForce {};
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

// used for residuum in SingleParticleKernel
template <typename T, uint64_t size>
struct cheapArray
{
    T data[size];
    //-----------------------------------------------------------------------------
    //! Access operator.
    //!
    //! \param index The index of the element to be accessed.
    //!
    //! Returns the requested element per reference.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T &operator[](uint64_t index)
    {
        return data[index];
    }
    //-----------------------------------------------------------------------------
    //! Access operator.
    //!
    //! \param index The index of the element to be accessed.
    //!
    //! Returns the requested element per constant reference.
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const T &operator[](uint64_t index) const
    {
        return data[index];
    }
};

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
    Element const & ts
)
-> Element
{
    //Element p=0.1; // 10%
    Element p=0.3; // 10%

    Element dv = vk;

    Element ln1p = log(1+p)/log(exp(1.0));

    Element restore = -ln1p/ts * particleMass;

    return restore * dv;
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
    std::size_t elems,
    std::size_t blockSize
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
        Element ts,
        double* residuum,
        std::size_t verletStep
    ) const
    {
        auto threadIndex  = alpaka::idx::getIdx<
            alpaka::Grid,
            alpaka::Threads
        >( acc )[ 0u ];

        auto threadBlockIndex  = alpaka::idx::getIdx<
            alpaka::Grid,
            alpaka::Blocks
        >( acc )[ 0u ];

        auto const start = threadIndex * elems;
        auto const   end = alpaka::math::min(
            acc,
            (threadIndex + 1) * elems,
            problemSize
        );

        auto const threadId(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

        // allocate shared memory for block-local residuum
        auto &localResiduum(
            alpaka::block::shared::st::allocVar<cheapArray<Element, blockSize>,
                                                    __COUNTER__>(acc));

        localResiduum[threadId] = 0.0;

        LLAMA_INDEPENDENT_DATA
        //for ( auto pos = start; pos < end; ++pos )
        for ( auto pos = start; pos < end; ++pos )
        {
            // cooling laser
            Element lForce[3] = {
                cooling_linear( particles( pos )( dd::Vel(), dd::X()),
                                ts),
                cooling_linear( particles( pos )( dd::Vel(), dd::Y()),
                                ts),
                cooling_linear( particles( pos )( dd::Vel(), dd::Z()),
                                ts),
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

            // verlet-integration part 1
            if ( verletStep == 0 ){
                particles( pos )( dd::Pos(), dd::X() ) +=
                    ts *
                    ( particles( pos )( dd::Vel(), dd::X() ) +
                      0.5 * ts * a[0] );
                particles( pos )( dd::Pos(), dd::Y() ) +=
                    ts *
                    ( particles( pos )( dd::Vel(), dd::Y() ) +
                      0.5 * ts * a[1] );
                particles( pos )( dd::Pos(), dd::Z() ) +=
                    ts *
                    ( particles( pos )( dd::Vel(), dd::Z() ) +
                      0.5 * ts * a[2] );

                particles( pos )( dd::Vel(), dd::X() ) +=
                    0.5 * ts * a[0];
                particles( pos )( dd::Vel(), dd::Y() ) +=
                    0.5 * ts * a[1];
                particles( pos )( dd::Vel(), dd::Z() ) +=
                    0.5 * ts * a[2];
            }
            // verlet-integration part 2
            if ( verletStep == 1 ){
                particles( pos )( dd::Vel(), dd::X() ) +=
                    0.5 * ts * a[0];
                particles( pos )( dd::Vel(), dd::Y() ) +=
                    0.5 * ts * a[1];
                particles( pos )( dd::Vel(), dd::Z() ) +=
                    0.5 * ts * a[2];
            }

            // reset coulomb forces
            particles( pos )( dd::CForce(), dd::X() )  = 0;
            particles( pos )( dd::CForce(), dd::Y() )  = 0;
            particles( pos )( dd::CForce(), dd::Z() )  = 0;

            // calculate residuum
            Element VelAbs = sqrt(
                particles( pos )( dd::Vel(), dd::X() ) *
                particles( pos )( dd::Vel(), dd::X() ) +
                particles( pos )( dd::Vel(), dd::Y() ) *
                particles( pos )( dd::Vel(), dd::Y() ) +
                particles( pos )( dd::Vel(), dd::Z() ) *
                particles( pos )( dd::Vel(), dd::Z() )
                    );

            localResiduum[threadId] = alpaka::math::max( acc, localResiduum[threadId], VelAbs );

        }

        syncBlockThreads(acc);

        auto n = NBODY_BLOCK_SIZE / 2;

        // these two versions do not work: we do not get block-individual output. Is this caused by the for-loops?! they seem to be executed by block 0 only...
        //while (n) {
        //    if (threadId < n && (threadId + n) < blockSize)
        //        //localResiduum[threadId] = alpaka::math::max(acc, localResiduum[threadId], localResiduum[threadId + n]);
        //        localResiduum[threadId] = threadId;

        //    n /= 2;

        //    syncBlockThreads(acc);
        //}
        //if (threadId == 0){
        //for ( int i=0; i<blockSize && i<(problemSize - threadBlockIndex * blockSize); i++){  
        //    //localResiduum[0] = alpaka::math::max( acc, localResiduum[threadId], localResiduum[threadId+1]);
        //    localResiduum[0] = threadBlockIndex;
        //    syncBlockThreads(acc);
        //}
        //}

        // this version works
        if (threadId == 0){
        int i=0;
        while(i<blockSize-1 && i<(problemSize - threadBlockIndex * blockSize)-1){
            //localResiduum[0] = i; // gives the correct number of threads used in each block
            //localResiduum[0] = threadBlockIndex; // gives correct block Id
            //localResiduum[0] = localResiduum[120]; 
            localResiduum[0] = alpaka::math::max( acc, localResiduum[0], localResiduum[i+1]);
            //localResiduum[0] = alpaka::math::max( acc, localResiduum[0], localResiduum[i]);
            i++; 
        }

        //if (threadId == 0)
            residuum[threadBlockIndex] = localResiduum[0];
            //residuum[threadBlockIndex] = threadId;
        }

        syncBlockThreads(acc);

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

    //if (myid == 0) {
    //    std::cout << (size * problemSize) / 1000 << " thousand particles (";
    //    std::cout << human_readable(size * (problemSize * llama::SizeOf<Particle>::value)) << ")\n";
    //}

    HRChrono chrono;

    particles.allocate(size * problemSize);
    auto   hostView = LocalFactory::allocView( mapping, particles.lbegin() );
    auto    devView =    DevFactory::allocView( mapping,  devAcc );
    auto mirrorView = MirrorFactory::allocView( mapping, devView );

    // will be used as double buffer for remote->host and host->device copying
    auto   remoteHostView =   HostFactory::allocView( mapping, devHost );
    auto    remoteDevView =    DevFactory::allocView( mapping,  devAcc );
    auto remoteMirrorView = MirrorFactory::allocView( mapping, devView );

    //chrono.printAndReset("Alloc:");

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

    //chrono.printAndReset("Init:");

    alpaka::mem::view::ViewPlainPtr<DevHost, unsigned char, Dim, Size> hostPlain(
        reinterpret_cast<unsigned char*>(particles.lbegin()), devHost, problemSize * llama::SizeOf<Particle>::value);
    alpaka::mem::view::copy(queue,
        devView.blob[0].buffer,
        hostPlain,
        problemSize * llama::SizeOf<Particle>::value);

    //chrono.printAndReset("Copy H->D");

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
        elemCount,
        blockSize
    > singleParticleKernel;

    // for residuum
    double* blockResiduum = new double[ blocks[0] ]; //each block has it's owm residuum

    auto hostMemory = alpaka::mem::buf::alloc<double, Size>(devHost, blocks[0] * sizeof(double));

    alpaka::mem::buf::Buf<DevAcc, double, Dim, Size> sourceDeviceMemory =
        alpaka::mem::buf::alloc<double, Size>(devAcc, blocks[0]*sizeof(double));

    alpaka::mem::view::copy(queue, sourceDeviceMemory, hostMemory, blocks[0] * sizeof(double));

    Allreduce globalResiduum( dash::Team::All() );

    auto unitResiduum = 0.0; // residuum for each unit

    double n = blocks[0] / 2;

    // for verlet integration
    std::size_t verletStep = 0;

    // initialize progress bar
    Element progress;
    //if ( myid == 0 ){
    //    std::cout << "Progress:\n";
    //    std::cout << "0%.....................50%...................100%\n";
    //    progress = 0.02;
    //}

    std::cout << DATA_DUMP_STEPS << std::endl;

    auto maxSteps = DATA_DUMP_STEPS;
    std::cout << "Steps, Residuum (Velocity in [m/s])\n";
    // start simulation
    std::size_t s = 0;
    //for ( std::size_t s = 0; s < steps; ++s)
    do
    {
        // TODO: // copy the data to the GPU
        // alpaka::mem::view::copy(queue, sourceDeviceMemory, hostMemory, n);

        //std::cout<<(Element)s/(Element)steps<<std::endl;
        //if ( ( (Element)s/(Element)steps ) >= progress && myid == 0 ){
        //    progress+=0.02;
        //    std::cout << "."<<std::flush;
        //}

        // dump data to file, print first and last step and in given interval
        /*
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
        */

        /* pair-wise with local particles */
        alpaka::kernel::exec< Acc > (
            queue,
            workdiv,
            particleInteractionKernel,
            mirrorView,
            mirrorView,
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
                mirrorView,
                remoteMirrorView,
                ts
            );

            //chrono.printAndReset("Update remote kernel:");

        }

        verletStep = s%2;
        // move kernel
        alpaka::kernel::exec<Acc>(
            queue,
            workdiv,
            singleParticleKernel,
            mirrorView,
            ts,
            alpaka::mem::view::getPtrNative(sourceDeviceMemory),
            verletStep
        );
        //chrono.printAndReset("Move kernel:         ");
        /* TODO: //  download result from GPU
        T resultGpuHost;
        auto resultGpuDevice =
        alpaka::mem::view::ViewPlainPtr<DevHost, T, Dim, Idx>(
        &resultGpuHost, devHost, static_cast<Extent>(blockSize));
        alpaka::mem::view::copy(queue, resultGpuDevice, destinationDeviceMemory, 1);
        */


        dummy( static_cast<void*>( mirrorView.blob[0] ) );

        alpaka::mem::view::copy(queue,
            hostPlain,
            devView.blob[0].buffer,
                problemSize * llama::SizeOf<Particle>::value);

        particles.barrier();

        // get residuum array back from kernel w/ number of elements = number of blocks
        alpaka::mem::view::copy(queue, hostMemory, sourceDeviceMemory, sizeof(double)*blocks[0]);
        blockResiduum = alpaka::mem::view::getPtrNative(hostMemory);

        

        // residuum from blocks
        if ( blocks[0] > 1){
            for (int id = 1; id < blocks[0]; id++){
                blockResiduum[0] = max(blockResiduum[0], blockResiduum[id]);
            }
        }
        unitResiduum = blockResiduum[0];

        /* res from this iteration */
        //globalResiduum.set( blockResiduum, dash::Team::All() );
        globalResiduum.set( &unitResiduum, dash::Team::All() );

        // calculate global residuum
        globalResiduum.collect_and_spread( dash::Team::All() );
        globalResiduum.wait( dash::Team::All() );

        //if ( s == 0 || s % DATA_DUMP_STEPS == 0 || s == (steps - 1) ){
        if ( s == 0 || s % maxSteps == 0 || s == (steps - 1) ){
            if ( myid==0 ){
                std::cout << s+1 <<  ", " << globalResiduum.get() << std::endl;
            }
        }

        ++s;
    } while ( ( double(globalResiduum.get()) > RESIDUUM || s < 20 ) && s < NBODY_STEPS ); // need s < 20 to get initial residuum > RESIDUUM

    // print final residuum
    if ( s % maxSteps != 0 ){
        if ( myid==0 ){
            std::cout << s+1 <<  ", " << globalResiduum.get() << std::endl;
        }
    }

    //std::cout<<std::endl; // at the end of progress bar

    //chrono.printAndReset("Copy D->H");

    dash::finalize();

    return 0;
}
