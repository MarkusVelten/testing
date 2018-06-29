#pragma once

#include <type_traits>

namespace internal
{

template<
	typename T_DstView,
	typename T_SrcView,
	typename T_Queue,
	typename T_SFINAE = void
>
struct AlpakaMemCopy
{
	void operator()(
		T_DstView & dstView,
		T_SrcView const & srcView
	)
	{
		std::cout << "Not implemented yet\n";
	}
};

template<
	typename T_DstView,
	typename T_SrcView,
	typename T_Queue
>
struct AlpakaMemCopy<
	T_DstView,
	T_SrcView,
	T_Queue,
	typename std::enable_if<
		std::is_same<
			typename T_DstView::Mapping,
			typename T_SrcView::Mapping
		>::value &&
		T_DstView::Mapping::blobCount == T_SrcView::Mapping::blobCount
	>::type
>
{
	void operator()(
		T_DstView & dstView,
		T_SrcView const & srcView,
		typename T_DstView::Mapping::UserDomain const userDomainSize,
		T_Queue & queue
	)
	{
		if (dstView.mapping.userDomainSize == srcView.mapping.userDomainSize)
		{
			for (std::size_t i = 0; i < T_DstView::Mapping::blobCount; ++i)
				alpaka::mem::view::copy(
					queue,
					dstView.blob[i].buffer,
					srcView.blob[i].buffer,
					alpaka::vec::Vec<
						alpaka::dim::Dim< decltype( dstView.blob[i].buffer ) >,
						alpaka::idx::Idx< decltype( dstView.blob[i].buffer ) >
					>( dstView.mapping.userDomainSize[0] )
				);
		}
		else
			std::cout << "Not implemented yet\n";
	}
};

} // namespace internal

template<
	typename T_DstView,
	typename T_SrcView,
	typename T_Queue
>
void alpakaMemCopy(
	T_DstView & dstView,
	T_SrcView const & srcView,
	typename T_DstView::Mapping::UserDomain const userDomainSize,
	T_Queue & queue
)
{
	internal::AlpakaMemCopy<
		T_DstView,
		T_SrcView,
		T_Queue
	>()(
		dstView,
		srcView,
		userDomainSize,
		queue
	);
}
