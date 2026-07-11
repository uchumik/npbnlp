include(FetchContent)
FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG v2.13.6
	)
FetchContent_MakeAvailable(pybind11)

include_directories(${CMAKE_SOURCE_DIR}/io/include)
include_directories(${CMAKE_SOURCE_DIR}/rd/include)
include_directories(${CMAKE_SOURCE_DIR}/lm/include)
include_directories(${CMAKE_SOURCE_DIR}/ma/include)
include_directories(${CMAKE_SOURCE_DIR}/tg/include)
include_directories(${CMAKE_SOURCE_DIR}/pa/include)
include_directories(${CMAKE_SOURCE_DIR}/sb/include)
include_directories(${CPLUS_INCLUDE_PATH})
link_directories(${CMAKE_SOURCE_DIR}/io/src)
link_directories(${CMAKE_SOURCE_DIR}/rd/src)
link_directories(${CMAKE_SOURCE_DIR}/lm/src)
link_directories(${CMAKE_SOURCE_DIR}/ma/src)
link_directories(${CMAKE_SOURCE_DIR}/sb/src)
link_directories(${CMAKE_SOURCE_DIR}/pa/src)
link_directories(${LIBRARY_PATH})

find_package(Boost COMPONENTS serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# nphsmm/nnpylm/ihmm (tg) are only compiled into the `ne` executable, not a
# shared library (see tests/CMakeLists.txt, which solved this the same way) -
# compile the sources directly into the module target.
pybind11_add_module(npbnlp_py
	src/binding.cc
	${CMAKE_SOURCE_DIR}/tg/src/nphsmm.cc
	${CMAKE_SOURCE_DIR}/tg/src/clattice.cc
	${CMAKE_SOURCE_DIR}/tg/src/nnpylm.cc
	${CMAKE_SOURCE_DIR}/tg/src/ihmm.cc
	${CMAKE_SOURCE_DIR}/tg/src/hlattice.cc
	)

set_target_properties(npbnlp_py PROPERTIES OUTPUT_NAME "npbnlp")

target_compile_features(npbnlp_py PUBLIC cxx_std_17)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
	target_link_libraries(npbnlp_py PUBLIC OpenMP::OpenMP_CXX io rd lm tn icuuc sb gr ${Boost_LIBRARIES})
else()
	target_link_libraries(npbnlp_py PUBLIC io rd lm tn icuuc sb gr ${Boost_LIBRARIES})
endif()

# libsb.so embeds usbd_lstm.cc whenever Torch was found locally (see
# sb/src/CMakeLists.txt), but that CMakeLists only links Torch into the
# `usbd` executable, not into the `sb` shared library itself - so libsb.so
# carries unresolved Torch symbols that only surface on eager dlopen (which
# is what Python's import does). Link Torch here too, scoped to our own
# target, rather than touching the existing sb/src/CMakeLists.txt.
find_package(Torch QUIET)
if (Torch_FOUND)
	target_link_libraries(npbnlp_py PUBLIC ${TORCH_LIBRARIES})
endif()

# for ICU
target_link_directories(npbnlp_py PUBLIC /usr/local/lib)
