#include <iostream>
#include <utility>

#include <typeinfo>

#include <libdash.h>
#include <dash/LocalMirror.h>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#define NBODY_CUDA 1
#else
#define NBODY_CUDA 0
#endif

#define NBODY_PROBLEM_SIZE 16*1024
#define NBODY_BLOCK_SIZE 256


#if BOOST_VERSION < 106700 && (__CUDACC__ || __IBMCPP__)
    #ifdef BOOST_PP_VARIADICS
        #undef BOOST_PP_VARIADICS
    #endif
    #define BOOST_PP_VARIADICS 1
#endif

#include <alpaka/alpaka.hpp>
#ifdef __CUDACC__
	#define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC __forceinline__
#else
	#define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC inline
#endif
#include <llama/llama.hpp>
#include <random>

#include <AlpakaAllocator.hpp>
#include <AlpakaMemCopy.hpp>


using Element = float;

// declare data structure
namespace dd
{
    struct Pos {};
    struct X {};
    struct Y {};
}

struct particle
{
    struct
    {
        Element x, y;
    } pos;
};

using Particle = llama::DS<
    llama::DE< dd::Pos, llama::DS<
        llama::DE< dd::X, Element >,
        llama::DE< dd::Y, Element >,
    > >
>;


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
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
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


    particles.allocate(size * problemSize);
    auto   hostView = LocalFactory::allocView( mapping, particles.lbegin() );
    auto    devView =    DevFactory::allocView( mapping,  devAcc );

    // will be used as double buffer for remote->host and host->device copying
//    auto   remoteHostView =   HostFactory::allocView( mapping, devHost );
    auto    remoteDevView =    DevFactory::allocView( mapping,  devAcc );


    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < problemSize; ++i)
    {
        hostView(i)( dd::Pos(), dd::X() ) = myid;

        hostView(i)( dd::Pos(), dd::Y() ) = myid;

    }

    alpaka::mem::view::ViewPlainPtr<DevHost, unsigned char, Dim, Size> hostPlain(
        reinterpret_cast<unsigned char*>(particles.lbegin()), devHost, problemSize * llama::SizeOf<Particle>::value);
    alpaka::mem::view::copy(queue,
        devView.blob[0].buffer,
        hostPlain,
        problemSize * llama::SizeOf<Particle>::value);


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

    auto const workdiv = alpaka::workdiv::WorkDivMembers<
        Dim,
        Size
    > {
        blocks,
        threads,
        elems
    };

    // copy hostView to devView

    /* pair-wise with remote particles */
    std::cout << "Copying data...\n";
    std::cout << "size = " << size << std::endl;
    for (dart_unit_t unit_it = 1; unit_it < size; ++unit_it)
    {
        std::cout << "myid =  " << myid << std::endl;
        dart_unit_t remote = (myid + unit_it) % size;

        // get remote local block into remoteHostView
        auto remote_begin = particles.begin() + (remote * problemSize);
        auto remote_end   = remote_begin + problemSize;

    // LocalMirror:
#if 1 == 1
        using mirror_t = dash::LocalMirror<decltype(particles.begin()), dash::HostSpace>;
        std::cout << "HostSpace\n";
#elif 1 == 2
        using mirror_t = dash::LocalMirror<decltype(particles.begin()), dash::HBWSpace>;
        std::cout << "HBWSpace\n";
#endif

        mirror_t mirror{};

        mirror.replicate(remote_begin, remote_end);
        
        auto remoteHostView = LocalFactory::allocView( mapping, mirror.begin() );

        if (remoteHostView(0)( dd::Pos(), dd::X() ) != remote)
          std::cout << "wrong!\n";  

    //    alpakaMemCopy( remoteDevView, remoteHostView, userDomainSize, queue );

    }

    dash::finalize();

    return 0;
}
