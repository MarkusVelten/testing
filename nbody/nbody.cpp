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

#define FACTOR 10

// (de)activate residuum-calculations for velocity & acceleration
#define SWITCH_RESIDUUM 0

#if SWITCH_RESIDUUM == 1
#define RESIDUUM 0.0001 // finish simulation at this residuum (max. velocity in [m/s])
#define DATA_DUMP_STEPS 10 * FACTOR  // write data to file every N steps
#endif

// (de)activate async copy
#define SWITCH_ASYNC 0

// (de)activate use of SLURM
#define SWITCH_SLURM 1

// (de)activate dump of results every DATA_DUMP_STEPS
#define SWITCH_DATA_DUMP 1
#if SWITCH_DATA_DUMP == 1
#define DATA_DUMP_STEPS 1 * FACTOR  // write data to file every N steps
#endif

#define NBODY_PROBLEM_SIZE 8
#define NBODY_BLOCK_SIZE 128
//#define NBODY_STEPS 100000 * FACTOR 
#define NBODY_STEPS 1 * FACTOR 

#define A 54059 /* a prime */
#define B 76963 /* another prime */
#define C 86969 /* yet another prime */
#define FIRSTH 37 /* also prime */
unsigned hash_str(const char* s, size_t length)
{
   unsigned h = FIRSTH;
   while (length) {
     h = (h * A) ^ (s[0] * B);
     s++;
     length--;
   }
   return h; // or return h % C;
}

using Element = double; // change to double if needed

constexpr Element EPS2 = 1e-10;

constexpr Element ts = 1e-7 / (1.0 * FACTOR); // timestep in [s]

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
    typename T_Acc,
    typename T_VirtualDatum1,
    typename T_VirtualDatum2
>
LLAMA_FN_HOST_ACC_INLINE
auto
pPInteraction(
    T_Acc const &acc,
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
    Element dist = alpaka::math::sqrt(acc, distSqr);
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
    typename T_Acc,
    typename T_VirtualDatum1
>
LLAMA_FN_HOST_ACC_INLINE
auto
cooling_linear(
    T_Acc const &acc,
    T_VirtualDatum1&& vk, //self
    Element const & ts
)
-> Element
{
    Element p=0.1; // 10%

    Element dv = vk;

    Element ln1p = alpaka::math::log(acc, (1+p)) /
                   alpaka::math::log(acc, alpaka::math::exp(acc, 1.0) );

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
    //std::size_t problemSize,
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
        Element ts,
        dart_unit_t myid,
        std::size_t problemSize
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

        auto blockIndex  = alpaka::idx::getIdx<
            alpaka::Grid,
            alpaka::Blocks
        >( acc )[ 0u ];

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
            //auto const start2 = blockIndex * blockSize;
            auto const   end2 = alpaka::math::min(
                acc,
                start2 + blockSize,
                problemSize
            );
            auto const   size2 = end2 - start2;

            LLAMA_INDEPENDENT_DATA
            for (
                auto pos2 = decltype(size2)(0);
                pos2 + threadBlockIndex < size2;
                pos2 += threads
            ){
                //temp(pos2 + threadBlockIndex ) = remoteParticles( start2 + pos2 + threadBlockIndex );
                temp(pos2 + threadBlockIndex ) = remoteParticles( start2 + pos2 + threadBlockIndex );
                //printf("%d, remoteParticlesX = %f\n",(int)myid, remoteParticles( start2 + pos2 + threadBlockIndex )(dd::Pos(), dd::X()));
                //printf("temp(%d + %d) = temp(%d) = remoteParticles(%d + %d + %d) = remoteParticles(%d); myid = %d; %d; %d\n", (int)pos2, (int)threadBlockIndex, (int)(pos2 + threadBlockIndex), (int)start2, (int)pos2, (int)threadBlockIndex, (int)(start2 + pos2 + threadBlockIndex), (int)myid, (int)(NBODY_PROBLEM_SIZE * myid + start2 + pos2 + threadBlockIndex), (int)b);
            }

            
            syncBlockThreads(acc);
            // compute loop
            LLAMA_INDEPENDENT_DATA
            for ( auto pos2 = decltype(size2)(0); pos2 < size2; ++pos2 ){
                LLAMA_INDEPENDENT_DATA
                for ( auto pos = start; pos < end; ++pos ){
                    pPInteraction(
                        acc,
                        localParticles( pos ),
                        temp( pos2 ),
                        ts
                    );
                    //printf("pos = %d; pos2+start2 = %d;  size2 = %d; myid = %d; local p = %d; remote p = %d\n",(int)pos, (int)(pos2+start2), (int)size2, (int)myid, (int)(NBODY_PROBLEM_SIZE * myid + pos), (int)(NBODY_PROBLEM_SIZE * myid + pos2 + start2));
                    printf("%d, localX = %f; remoteX = %f\n", (int)myid, localParticles(pos)( dd::Pos(), dd::X() ),  temp(pos2)( dd::Pos(), dd::X() ) );

                }   
            }
            syncBlockThreads(acc);
        }
    }
};

template<
    //std::size_t problemSize,
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
#if SWITCH_RESIDUUM == 1
        double* velResiduum,
        double* accResiduum,
#endif
        std::size_t verletStep,
        std::size_t problemSize
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

#if SWITCH_RESIDUUM == 1
        auto const threadId(alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);

        // allocate shared memory for block-local residuum
        // local velocity residuum
        auto &velLocalResiduum(
            alpaka::block::shared::st::allocVar<cheapArray<Element, blockSize>,
                                                    __COUNTER__>(acc));
        velLocalResiduum[threadId] = 0.0;

        // local acceleration residuum
        auto &accLocalResiduum(
            alpaka::block::shared::st::allocVar<cheapArray<Element, blockSize>,
                                                    __COUNTER__>(acc));

        accLocalResiduum[threadId] = 0.0;
#endif

        LLAMA_INDEPENDENT_DATA
        for ( auto pos = start; pos < end; ++pos )
        {
            // cooling laser
            Element lForce[3] = {
                cooling_linear( acc,
                                particles( pos )( dd::Vel(), dd::X()),
                                ts),
                cooling_linear( acc,
                                particles( pos )( dd::Vel(), dd::Y()),
                                ts),
                cooling_linear( acc,
                                particles( pos )( dd::Vel(), dd::Z()),
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
            particles( pos )( dd::CForce() ) = 0;

#if SWITCH_RESIDUUM == 1
            // calculate residuum
            Element VelAbs = alpaka::math::sqrt( acc,
                particles( pos )( dd::Vel(), dd::X() ) *
                particles( pos )( dd::Vel(), dd::X() ) +
                particles( pos )( dd::Vel(), dd::Y() ) *
                particles( pos )( dd::Vel(), dd::Y() ) +
                particles( pos )( dd::Vel(), dd::Z() ) *
                particles( pos )( dd::Vel(), dd::Z() )
                    );
            Element AccAbs = alpaka::math::sqrt( acc,
                a[0] * a[0] +
                a[1] * a[1] +
                a[2] * a[2] );

            velLocalResiduum[threadId] = alpaka::math::max( acc, velLocalResiduum[threadId], VelAbs );
            accLocalResiduum[threadId] = alpaka::math::max( acc, accLocalResiduum[threadId], AccAbs );
#endif

        }

        syncBlockThreads(acc);

#if SWITCH_RESIDUUM == 1
        auto n = NBODY_BLOCK_SIZE / 2;

        //while (n) {
        //    if (threadId < n && (threadId + n) < blockSize)
        //        //accLocalResiduum[threadId] = alpaka::math::max(acc, localResiduum[threadId], localResiduum[threadId + n]);
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

        if (threadId == 0){
            int i=0;
            while(i<blockSize-1 && i<(problemSize - threadBlockIndex * blockSize)-1){
                //localResiduum[0] = i; // gives the correct number of threads used in each block
                //localResiduum[0] = threadBlockIndex; // gives correct block Id
                //localResiduum[0] = localResiduum[120]; 
                velLocalResiduum[0] = alpaka::math::max( acc, velLocalResiduum[0], velLocalResiduum[i+1]);
                accLocalResiduum[0] = alpaka::math::max( acc, accLocalResiduum[0], accLocalResiduum[i+1]);
                //localResiduum[0] = alpaka::math::max( acc, localResiduum[0], localResiduum[i]);
                i++; 
            }

        //if (threadId == 0)
            velResiduum[threadBlockIndex] = velLocalResiduum[0];
            accResiduum[threadBlockIndex] = accLocalResiduum[0];
            //residuum[threadBlockIndex] = threadId;
        }

        syncBlockThreads(acc);
#endif

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
    std::cout.precision(12);
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
#if SWITCH_SLURM == 1
    DevAcc const devAcc( alpaka::pltf::getDevByIdx< PltfAcc >( atoi( getenv( "SLURM_LOCALID" ) ) ) );
#else
    DevAcc const devAcc( alpaka::pltf::getDevByIdx< PltfAcc >( 0u ) );
#endif
    DevHost const devHost( alpaka::pltf::getDevByIdx< PltfHost >( 0u ) );

    dash::init(&argc, &argv);

    dart_unit_t myid = dash::myid();
    dart_unit_t size = dash::size();

    Queue queue( devAcc ) ;

    // NBODY
    constexpr std::size_t problemSize = NBODY_PROBLEM_SIZE;
    std::size_t localProblemSize = problemSize / (std::size_t)size; // problemSize for each unit
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
    const UserDomain userDomainSize{ localProblemSize };

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
        std::cout << problemSize / 1000 << " thousand particles (";
        std::cout << human_readable((problemSize * llama::SizeOf<Particle>::value)) << ")\n";
    }

    HRChrono chrono;

    particles.allocate(problemSize);
    auto   hostView = LocalFactory::allocView( mapping, particles.lbegin() );
    auto    accView =    DevFactory::allocView( mapping,  devAcc );
    auto mirrorView = MirrorFactory::allocView( mapping, accView );

    // will be used as double buffer for remote->host and host->device copying
    auto   remoteHostView =   HostFactory::allocView( mapping, devHost );
    auto    remoteAccView =    DevFactory::allocView( mapping,  devAcc );
    auto remoteMirrorView = MirrorFactory::allocView( mapping, remoteAccView );

    //chrono.printAndReset("Alloc:");

    std::default_random_engine generator;
    //std::minstd_rand0 generator(1);
    std::normal_distribution< Element > distribution(
        Element( 0 ), // mean
        Element( 5e-4 )  // stddev
    );

    auto random = 0.0;
    
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < (localProblemSize * myid * 3); ++i)
    {
        random = distribution(generator);
    }

    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < localProblemSize; ++i)
    {
        // initialize position in X, Y, Z with random
        random = distribution(generator);
        hostView(i)(dd::Pos(), dd::X()) = random;

        random = distribution(generator);
        hostView(i)(dd::Pos(), dd::Y()) = random;

        random = distribution(generator);
        hostView(i)(dd::Pos(), dd::Z()) = random;

        // initialize velocity in X, Y, Z with zero (approximation)
        hostView(i)(dd::Vel()) = 0;

        // initialize coulomb force in X, Y, Z with zero
        hostView(i)(dd::CForce()) = 0;

        
    }

    //for (std::size_t o = 0; o < size; o++){
    //    if (myid == o){
    //        for (std::size_t i = 0; i < localProblemSize; ++i)
    //        {
    //        std::cout << hostView(i)(dd::Pos(), dd::X()) \
    //            << "\t" \
    //            << hostView(i)(dd::Pos(), dd::Y()) \
    //            << "\t" \
    //            << hostView(i)(dd::Pos(), dd::Z()) \
    //            << std::endl;
    //        }
    //    }
    //}

    //chrono.printAndReset("Init:");

    
    alpaka::mem::view::ViewPlainPtr<DevHost, unsigned char, Dim, Size> hostPlain(
        reinterpret_cast<unsigned char*>(particles.lbegin()), devHost, localProblemSize * llama::SizeOf<Particle>::value);
    alpaka::mem::view::copy(queue,
        accView.blob[0].buffer,
        hostPlain,
        localProblemSize * llama::SizeOf<Particle>::value);

    std::cout << myid << ":0 " << hash_str(reinterpret_cast<char*>(alpaka::mem::view::getPtrNative(hostPlain)), localProblemSize * llama::SizeOf<Particle>::value) << std::endl;
    //chrono.printAndReset("Copy H->D");

    const alpaka::vec::Vec< Dim, Size > elems (
        static_cast< Size >( elemCount )
    );
    const alpaka::vec::Vec< Dim, Size > threads (
        static_cast< Size >( threadCount )
    );
    constexpr auto innerCount = elemCount * threadCount;
    const alpaka::vec::Vec< Dim, Size > blocks (
        static_cast< Size >( ( localProblemSize + innerCount - 1 ) / innerCount )
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

    // copy hostView to accView
    ParticleInteractionKernel<
        //localProblemSize,
        elemCount,
        blockSize
    > particleInteractionKernel;
    SingleParticleKernel<
        //problemSize/size,
        elemCount,
        blockSize
    > singleParticleKernel;

    // for residuum
#if SWITCH_RESIDUUM == 1
    // velocity residuum
    double* velBlockResiduum = new double[ blocks[0] ]; //each block has it's own residuum

    auto velHostMemory = alpaka::mem::buf::alloc<double, Size>(devHost, blocks[0] * sizeof(double));

    alpaka::mem::buf::Buf<DevAcc, double, Dim, Size> velSourceDeviceMemory =
        alpaka::mem::buf::alloc<double, Size>(devAcc, blocks[0]*sizeof(double));

    alpaka::mem::view::copy(queue, velSourceDeviceMemory, velHostMemory, blocks[0] * sizeof(double));

    Allreduce velGlobalResiduum( dash::Team::All() );

    auto velUnitResiduum = 0.0; // residuum for each unit

    // acceleration residuum
    double* accBlockResiduum = new double[ blocks[0] ]; //each block has it's own residuum

    auto accHostMemory = alpaka::mem::buf::alloc<double, Size>(devHost, blocks[0] * sizeof(double));

    alpaka::mem::buf::Buf<DevAcc, double, Dim, Size> accSourceDeviceMemory =
        alpaka::mem::buf::alloc<double, Size>(devAcc, blocks[0]*sizeof(double));

    alpaka::mem::view::copy(queue, accSourceDeviceMemory, velHostMemory, blocks[0] * sizeof(double));

    Allreduce accGlobalResiduum( dash::Team::All() );

    auto accUnitResiduum = 0.0; // residuum for each unit

    double n = blocks[0] / 2;
#endif

    // for verlet integration
    std::size_t verletStep = 0;

    auto maxSteps = NBODY_STEPS;
    
#if SWITCH_DATA_DUMP == 1
    auto dumpSteps = DATA_DUMP_STEPS;
#endif
    
    // initialize progress bar
    //Element progress;
    //if ( myid == 0 ){
    //    std::cout << "Progress:\n";
    //    std::cout << "0%.....................50%...................100%\n";
    //    progress = 0.02;
    //}


    
    //if (myid == 0)
    //    std::cout << "Steps \t Residuum (Velocity in [m/s]) \t Residuum (Acceleration in [m/s^2]) \t pos(X) \t pos(Y) \t pos(Z) \n";

    // start simulation
    std::size_t s = 0;
    do
    {
        // alpaka::mem::view::copy(queue, velSourceDeviceMemory, velHostMemory, n);

        //if ( ( (Element)s/(Element)steps ) >= progress && myid == 0 ){
        //    progress+=0.02;
        //    std::cout << "."<<std::flush;
        //}

#if SWITCH_DATA_DUMP == 1
        // dump data to file, print first and last step and in given interval
        if ( s == 0 || s % dumpSteps == 0 || s == (steps - 1) ){
            std::string i = std::to_string(s);
            std::string fileName = "data";
            fileName.append(i);
            fileName.append(".");
            std::string j = std::to_string(myid);
            fileName.append(j);
            fileName.append(".csv");
            std::fstream myfile;
            //if ( myid == 0 ){
                myfile.open(fileName, std::fstream::out | std::fstream::trunc);

                myfile << "Residuum (Velocity in [m/s]),Residuum (Acceleration in [m/s^2]),pos(X),pos(Y),pos(Z) \n";
                myfile.close();
            //}
                // write local particles to file
            // for( std::size_t u = 0; u < size; u++){
             //   if ( myid == u){
                    std::ofstream fout;
                    std::ifstream fin;
                    fin.open(fileName);
                    fout.open(fileName, std::ios::app);
                    if(fin.is_open()){
                for (std::size_t i = 0; i < localProblemSize; ++i)
                {
                    fout << hostView(i)(dd::Pos(), dd::X()) \
                        << "," \
                        << hostView(i)(dd::Pos(), dd::Y()) \
                        << "," \
                        << hostView(i)(dd::Pos(), dd::Z()) \
                        << std::endl;
                }
                }
                    fin.close();
                    fout.close(); // Closing the file))
                //}
            //}
            //    // write remote particle positions to file
            //    if ( size > 1 ){
            //        for (std::size_t i = 0; i < problemSize; ++i)
            //        {
            //            myfile << velGlobalResiduum.get() \
            //                << "," \
            //                << accGlobalResiduum.get() \
            //                << "," \
            //                << remoteHostView(i)(dd::Pos(), dd::X()) \
            //                << "," \
            //                << remoteHostView(i)(dd::Pos(), dd::Y()) \
            //                << "," \
            //                << remoteHostView(i)(dd::Pos(), dd::Z()) \
            //                << std::endl;
            //        }
            //    }
                //myfile.close();
        }
        //}
#endif

#if SWITCH_ASYNC == 1       
            dart_unit_t remote = (myid + 1) % size;

            // get remote local block into remoteHostView 
            auto remote_begin = particles.begin() + (remote * localProblemSize);
            auto remote_end   = remote_begin + localProblemSize;
            auto target_begin = reinterpret_cast<particle*>(alpaka::mem::view::getPtrNative(remoteHostView.blob[0].buffer));
            auto fut_dest_end = dash::copy_async(remote_begin, remote_end, target_begin); // copy particles from remote
#endif
        /* pair-wise with local particles */
        alpaka::kernel::exec< Acc > (
            queue,
            workdiv,
            particleInteractionKernel,
            mirrorView,
            mirrorView,
            ts,
            myid,
            localProblemSize
        );

        //chrono.printAndReset("Update kernel:       ");

        /* pair-wise with remote particles */
        for (dart_unit_t unit_it = 1; unit_it < size; ++unit_it)
        {

#if SWITCH_ASYNC == 1
            auto copy_dest_end = fut_dest_end.get();

            alpakaMemCopy( remoteAccView, remoteHostView, userDomainSize, queue );

            remote = (myid + unit_it + 1) % size;

            if (remote != myid){
                // get remote local block into remoteHostView
                remote_begin = particles.begin() + (remote * localProblemSize);
                remote_end   = remote_begin + localProblemSize;
                target_begin = reinterpret_cast<particle*>(alpaka::mem::view::getPtrNative(remoteHostView.blob[0].buffer));
                fut_dest_end = dash::copy_async(remote_begin, remote_end, target_begin); // copy particles from remote
            }
#else
            dart_unit_t remote = (myid + unit_it) % size;

            // get remote local block into remoteHostView
            auto remote_begin = particles.begin() + (remote * localProblemSize);
            auto remote_end   = remote_begin + localProblemSize;
            auto target_begin = reinterpret_cast<particle*>(alpaka::mem::view::getPtrNative(remoteHostView.blob[0].buffer));

            particles.barrier();
            dash::copy(remote_begin, remote_end, target_begin); // copy particles from remote

            alpakaMemCopy( remoteAccView, remoteHostView, userDomainSize, queue );
#endif
            particles.barrier();

            alpaka::kernel::exec< Acc > (
                queue,
                workdiv,
                particleInteractionKernel,
                mirrorView,
                remoteMirrorView,
                ts,
                myid,
                localProblemSize
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
#if SWITCH_RESIDUUM == 1
            alpaka::mem::view::getPtrNative(velSourceDeviceMemory),
            alpaka::mem::view::getPtrNative(accSourceDeviceMemory),
#endif
            verletStep,
            localProblemSize
        );
        //chrono.printAndReset("Move kernel:         ");
        /* TODO: //  download result from GPU
        T resultGpuHost;
        auto resultGpuDevice =
        alpaka::mem::view::ViewPlainPtr<DevHost, T, Dim, Idx>(
        &resultGpuHost, devHost, static_cast<Extent>(blockSize));
        alpaka::mem::view::copy(queue, resultGpuDevice, destinationDeviceMemory, 1);
        */


        particles.barrier();
        dummy( static_cast<void*>( mirrorView.blob[0] ) );

        alpaka::mem::view::copy(queue,
            hostPlain,
            accView.blob[0].buffer,
                localProblemSize * llama::SizeOf<Particle>::value);

        particles.barrier();

#if SWITCH_RESIDUUM == 1
        // get residuum array back from kernel w/ number of elements = number of blocks
        alpaka::mem::view::copy(queue, velHostMemory, velSourceDeviceMemory, sizeof(double)*blocks[0]);
        velBlockResiduum = alpaka::mem::view::getPtrNative(velHostMemory);

        alpaka::mem::view::copy(queue, accHostMemory, accSourceDeviceMemory, sizeof(double)*blocks[0]);
        accBlockResiduum = alpaka::mem::view::getPtrNative(accHostMemory);

        // residuum from blocks
        if ( blocks[0] > 1){
            for (int id = 1; id < blocks[0]; id++){
                velBlockResiduum[0] = max(velBlockResiduum[0], velBlockResiduum[id]);
                accBlockResiduum[0] = max(accBlockResiduum[0], accBlockResiduum[id]);
            }
        }
        velUnitResiduum = velBlockResiduum[0];
        accUnitResiduum = accBlockResiduum[0];

        /* res from this iteration */
        velGlobalResiduum.set( &velUnitResiduum, dash::Team::All() );
        accGlobalResiduum.set( &accUnitResiduum, dash::Team::All() );

        // calculate global residuum
        velGlobalResiduum.collect_and_spread( dash::Team::All() );
        velGlobalResiduum.wait( dash::Team::All() );

        accGlobalResiduum.collect_and_spread( dash::Team::All() );
        accGlobalResiduum.wait( dash::Team::All() );

        
        if ( s == 0 || s % dumpSteps == 0 || s == (steps - 1) ){
            if ( myid==0 ){
                std::cout << s+1 <<  "\t" << velGlobalResiduum.get() << "\t" \// << std::endl;
                        << accGlobalResiduum.get() << "\t" \
                        << hostView(0)(dd::Pos(), dd::X()) << "\t" \
                        << hostView(0)(dd::Pos(), dd::Y()) << "\t" \
                        << hostView(0)(dd::Pos(), dd::Z()) \
                        << std::endl;
            }
        }
#endif 

        ++s;
#if SWITCH_RESIDUUM == 1
    } while ( ( double(velGlobalResiduum.get()) > RESIDUUM || s < 20 ) && s < maxSteps ); // need s < 20 to get initial residuum > RESIDUUM
#else
    } while ( s < maxSteps ); 
#endif

#if SWITCH_RESIDUUM == 1
    // print final residuum
    if ( s % maxSteps != 0 ){
        if ( myid==0 ){
            std::cout << s+1 <<  "\t" << velGlobalResiduum.get() << "\t" << accGlobalResiduum.get() << std::endl;
        }
    }
#endif

    //std::cout<<std::endl; // at the end of progress bar

    //chrono.printAndReset("Copy D->H");

    dash::finalize();

    return 0;
}
