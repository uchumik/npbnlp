include_directories(${CMAKE_SOURCE_DIR}/io/include)
include_directories(${CMAKE_SOURCE_DIR}/rd/include)
include_directories(${CMAKE_SOURCE_DIR}/lm/include)
include_directories(${CMAKE_SOURCE_DIR}/ma/include)
include_directories(${CMAKE_SOURCE_DIR}/tg/include)
include_directories(${CPLUS_INCLUDE_PATH})
link_directories(${CMAKE_SOURCE_DIR}/io/src)
link_directories(${CMAKE_SOURCE_DIR}/rd/src)
link_directories(${CMAKE_SOURCE_DIR}/lm/src)
link_directories(${CMAKE_SOURCE_DIR}/ma/src)
link_directories(${CMAKE_SOURCE_DIR}/tg/src)
link_directories(${LIBRARY_PATH})

find_package(Boost COMPONENTS serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

find_package(OpenMP)

add_executable(test_io test_io.cc)
add_executable(test_hpyp_symmetry test_hpyp_symmetry.cc)
add_executable(test_hpyp_roundtrip test_hpyp_roundtrip.cc)
add_executable(test_npylm_smoke test_npylm_smoke.cc)
add_executable(test_serialize_compat
	test_serialize_compat.cc
	${CMAKE_SOURCE_DIR}/tg/src/nphsmm.cc
	${CMAKE_SOURCE_DIR}/tg/src/clattice.cc
	${CMAKE_SOURCE_DIR}/tg/src/nnpylm.cc
	${CMAKE_SOURCE_DIR}/tg/src/ihmm.cc
	${CMAKE_SOURCE_DIR}/tg/src/hlattice.cc
	)

foreach(t test_io test_hpyp_symmetry test_hpyp_roundtrip test_npylm_smoke test_serialize_compat)
	target_compile_options(${t} PUBLIC
		$<$<CONFIG:Release>:-O2 -std=c++17>
		$<$<CONFIG:Debug>:-g -std=c++17>
		)
	target_link_directories(${t} PUBLIC /usr/local/lib ${LIBRARY_PATH})
endforeach()

if (OpenMP_CXX_FOUND)
	target_link_libraries(test_io PUBLIC OpenMP::OpenMP_CXX io icuuc)
	target_link_libraries(test_hpyp_symmetry PUBLIC OpenMP::OpenMP_CXX io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_hpyp_roundtrip PUBLIC OpenMP::OpenMP_CXX io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_npylm_smoke PUBLIC OpenMP::OpenMP_CXX io rd lm tn icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_serialize_compat PUBLIC OpenMP::OpenMP_CXX io rd lm tn icuuc ${Boost_LIBRARIES})
else()
	target_link_libraries(test_io PUBLIC io icuuc)
	target_link_libraries(test_hpyp_symmetry PUBLIC io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_hpyp_roundtrip PUBLIC io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_npylm_smoke PUBLIC io rd lm tn icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_serialize_compat PUBLIC io rd lm tn icuuc ${Boost_LIBRARIES})
endif()

add_test(NAME test_io COMMAND test_io ${CMAKE_CURRENT_SOURCE_DIR}/data/tiny_ja.txt)
add_test(NAME test_hpyp_symmetry COMMAND test_hpyp_symmetry)
add_test(NAME test_hpyp_roundtrip COMMAND test_hpyp_roundtrip ${CMAKE_CURRENT_SOURCE_DIR}/data/tiny_ja.txt)
add_test(NAME test_npylm_smoke COMMAND test_npylm_smoke ${CMAKE_CURRENT_SOURCE_DIR}/data/tiny_ja.txt)
add_test(NAME test_serialize_compat COMMAND test_serialize_compat)
