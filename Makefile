spmm:
	icpc -Wno-write-strings -g -std=c++0x -O3 -qopenmp -lpapi -vec-report1 -restrict -pthread  -march=native -Denable_gpu -Denable_mkl -m64 -w -O3 -parallel -qopenmp -Ofast -xCORE-AVX2  spmm_istream.cc -o spmm_istream -I /opt/software/papi/5.4.1/include
	icpc -Wno-write-strings -g -std=c++0x -O3 -qopenmp -lpapi -vec-report1 -restrict -pthread  -march=native -Denable_gpu -Denable_mkl -m64 -w -O3 -parallel -qopenmp -Ofast -xCORE-AVX2  spmm_istream_knl.cc -o spmm_istream_knl -I /opt/software/papi/5.4.1/include




xeon:
	icpc -Wno-write-strings -g -std=c++0x -O3 -qopenmp -lpapi -vec-report1 -restrict -pthread  -march=native -Denable_gpu -Denable_mkl -m64 -w -O3 -parallel -qopenmp -Ofast -xCORE-AVX2  sddmm_istream.cc -o sddmm_istream -I /opt/software/papi/5.4.1/include

knl:
	icpc -Wno-write-strings -g -std=c++0x -O3 -qopenmp -lpapi -vec-report1 -restrict -pthread  -march=native -Denable_gpu -Denable_mkl -m64 -w -O3 -parallel -qopenmp -Ofast -xCORE-AVX2  sddmm_istream_knl.cc -o sddmm_istream_knl -I /opt/software/papi/5.4.1/include
