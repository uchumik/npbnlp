include_directories(${CMAKE_SOURCE_DIR}/io/include)
include_directories(${CMAKE_SOURCE_DIR}/rd/include)
include_directories(${CMAKE_SOURCE_DIR}/lm/include)
include_directories(${CMAKE_SOURCE_DIR}/ma/include)
include_directories(${CMAKE_SOURCE_DIR}/tg/include)
include_directories(${CMAKE_SOURCE_DIR}/pa/include)
include_directories(${CPLUS_INCLUDE_PATH})
link_directories(${CMAKE_SOURCE_DIR}/io/src)
link_directories(${CMAKE_SOURCE_DIR}/rd/src)
link_directories(${CMAKE_SOURCE_DIR}/lm/src)
link_directories(${CMAKE_SOURCE_DIR}/ma/src)
link_directories(${CMAKE_SOURCE_DIR}/tg/src)
link_directories(${CMAKE_SOURCE_DIR}/pa/src)
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
add_executable(test_nphsmm_roundtrip
	test_nphsmm_roundtrip.cc
	${CMAKE_SOURCE_DIR}/tg/src/nphsmm.cc
	${CMAKE_SOURCE_DIR}/tg/src/clattice.cc
	${CMAKE_SOURCE_DIR}/tg/src/nnpylm.cc
	${CMAKE_SOURCE_DIR}/tg/src/ihmm.cc
	${CMAKE_SOURCE_DIR}/tg/src/hlattice.cc
	)
add_executable(test_snpylm_roundtrip
	test_snpylm_roundtrip.cc
	${CMAKE_SOURCE_DIR}/tg/src/snpylm.cc
	${CMAKE_SOURCE_DIR}/tg/src/clattice.cc
	)
add_executable(test_snpylm_infer
	test_snpylm_infer.cc
	${CMAKE_SOURCE_DIR}/tg/src/snpylm.cc
	${CMAKE_SOURCE_DIR}/tg/src/clattice.cc
	)
add_executable(test_ipcfg_span test_ipcfg_span.cc)
add_executable(test_ipcfg_split test_ipcfg_split.cc)
add_executable(test_ipcfg_wngram test_ipcfg_wngram.cc)
add_executable(test_negative_binomial
	test_negative_binomial.cc
	${CMAKE_SOURCE_DIR}/rd/src/negative_binomial.cc
	)

foreach(t test_io test_hpyp_symmetry test_hpyp_roundtrip test_npylm_smoke test_serialize_compat test_nphsmm_roundtrip test_snpylm_roundtrip test_snpylm_infer test_ipcfg_span test_ipcfg_split test_ipcfg_wngram test_negative_binomial)
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
	target_link_libraries(test_nphsmm_roundtrip PUBLIC OpenMP::OpenMP_CXX io rd lm tn icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_snpylm_roundtrip PUBLIC OpenMP::OpenMP_CXX io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_snpylm_infer PUBLIC OpenMP::OpenMP_CXX io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_ipcfg_span PUBLIC OpenMP::OpenMP_CXX gr io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_ipcfg_split PUBLIC OpenMP::OpenMP_CXX gr io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_ipcfg_wngram PUBLIC OpenMP::OpenMP_CXX gr io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_negative_binomial PUBLIC OpenMP::OpenMP_CXX)
else()
	target_link_libraries(test_io PUBLIC io icuuc)
	target_link_libraries(test_hpyp_symmetry PUBLIC io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_hpyp_roundtrip PUBLIC io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_npylm_smoke PUBLIC io rd lm tn icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_serialize_compat PUBLIC io rd lm tn icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_nphsmm_roundtrip PUBLIC io rd lm tn icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_snpylm_roundtrip PUBLIC io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_snpylm_infer PUBLIC io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_ipcfg_span PUBLIC gr io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_ipcfg_split PUBLIC gr io rd lm icuuc ${Boost_LIBRARIES})
	target_link_libraries(test_ipcfg_wngram PUBLIC gr io rd lm icuuc ${Boost_LIBRARIES})
endif()

add_test(NAME test_io COMMAND test_io ${CMAKE_CURRENT_SOURCE_DIR}/data/tiny_ja.txt)
add_test(NAME test_hpyp_symmetry COMMAND test_hpyp_symmetry)
add_test(NAME test_hpyp_roundtrip COMMAND test_hpyp_roundtrip ${CMAKE_CURRENT_SOURCE_DIR}/data/tiny_ja.txt)
add_test(NAME test_npylm_smoke COMMAND test_npylm_smoke ${CMAKE_CURRENT_SOURCE_DIR}/data/tiny_ja.txt)
add_test(NAME test_serialize_compat COMMAND test_serialize_compat)
add_test(NAME test_nphsmm_roundtrip COMMAND test_nphsmm_roundtrip)
add_test(NAME test_snpylm_roundtrip COMMAND test_snpylm_roundtrip)
add_test(NAME test_snpylm_infer COMMAND test_snpylm_infer)
add_test(NAME test_ipcfg_span COMMAND test_ipcfg_span ${CMAKE_CURRENT_SOURCE_DIR}/data/ipcfg_span_smoke.txt)
add_test(NAME test_ipcfg_split COMMAND test_ipcfg_split ${CMAKE_CURRENT_SOURCE_DIR}/data/ipcfg_span_smoke.txt)
add_test(NAME test_ipcfg_wngram COMMAND test_ipcfg_wngram ${CMAKE_CURRENT_SOURCE_DIR}/data/ipcfg_span_smoke.txt)
add_test(NAME test_negative_binomial COMMAND test_negative_binomial)
