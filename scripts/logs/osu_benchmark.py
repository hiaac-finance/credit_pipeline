# hpccm OSU Benchmark Recipe
Stage0 += baseimage(image='ubuntu:22.04')

# Install Mellanox OFED (adjust the version)
Stage0 += shell(commands=[
    'wget http://content.mellanox.com/ofed/MLNX_OFED_LINUX-5.8-3.0.7.0-ubuntu22.04-x86_64.tgz',
    'tar -xvzf MLNX_OFED_LINUX-5.8-3.0.7.0-ubuntu22.04-x86_64.tgz',
    'cd MLNX_OFED_LINUX-5.8-3.0.7.0-ubuntu22.04-x86_64 && ./mlnxofedinstall --without-dkms --all',
    '/etc/init.d/openibd restart'
])

# Install OpenMPI 4.1.1
Stage0 += shell(commands=[
    'wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz',
    'tar -xvzf openmpi-4.1.1.tar.gz',
    'cd openmpi-4.1.1 && ./configure --prefix=/usr/local && make all install',
    'ldconfig'
])

# Install GCC 12.3
Stage0 += gnu(fortran=True, version='12.3')

# Download and build OSU Micro-Benchmarks 7.4
Stage0 += shell(commands=[
    'wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.4.tar.gz',
    'tar -xvzf osu-micro-benchmarks-7.4.tar.gz',
    'cd osu-micro-benchmarks-7.4 && ./configure CC=mpicc CXX=mpicxx && make'
])

# Set environment variables
Stage0 += environment(variables={
    'PATH': '/usr/local/bin:$PATH',
    'LD_LIBRARY_PATH': '/usr/local/lib:$LD_LIBRARY_PATH'
})
