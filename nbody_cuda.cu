#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <chrono>

//double G = 6.674*std::pow(10,-11);
//double G = 1;

struct simulation {
  size_t nbpart;
  
  std::vector<double> mass;

  //position
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;

  //velocity
  std::vector<double> vx;
  std::vector<double> vy;
  std::vector<double> vz;

  //force
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;

  
  simulation(size_t nb)
    :nbpart(nb), mass(nb),
     x(nb), y(nb), z(nb),
     vx(nb), vy(nb), vz(nb),
     fx(nb), fy(nb), fz(nb) 
  {}
};

void random_init(simulation& s) {
  std::random_device rd;  
  std::mt19937 gen(rd());
  std::uniform_real_distribution dismass(0.9, 1.);
  std::normal_distribution dispos(0., 1.);
  std::normal_distribution disvel(0., 1.);

  for (size_t i = 0; i<s.nbpart; ++i) {
    s.mass[i] = dismass(gen);

    s.x[i] = dispos(gen);
    s.y[i] = dispos(gen);
    s.z[i] = dispos(gen);
    s.z[i] = 0.;
    
    s.vx[i] = disvel(gen);
    s.vy[i] = disvel(gen);
    s.vz[i] = disvel(gen);
    s.vz[i] = 0.;
    s.vx[i] = s.y[i]*1.5;
    s.vy[i] = -s.x[i]*1.5;
  }
  return;
  //normalize velocity (using normalization found on some physicis blog)
  double meanmass = 0;
  double meanmassvx = 0;
  double meanmassvy = 0;
  double meanmassvz = 0;
  for (size_t i = 0; i<s.nbpart; ++i) {
    meanmass += s.mass[i];
    meanmassvx += s.mass[i] * s.vx[i];
    meanmassvy += s.mass[i] * s.vy[i];
    meanmassvz += s.mass[i] * s.vz[i];
  }
  for (size_t i = 0; i<s.nbpart; ++i) {
    s.vx[i] -= meanmassvx/meanmass;
    s.vy[i] -= meanmassvy/meanmass;
    s.vz[i] -= meanmassvz/meanmass;
  }
}

void init_solar(simulation& s) {
  enum Planets {SUN, MERCURY, VENUS, EARTH, MARS, JUPITER, SATURN, URANUS, NEPTUNE, MOON};
  s = simulation(10);

  // Masses in kg
  s.mass[SUN] = 1.9891 * std::pow(10, 30);
  s.mass[MERCURY] = 3.285 * std::pow(10, 23);
  s.mass[VENUS] = 4.867 * std::pow(10, 24);
  s.mass[EARTH] = 5.972 * std::pow(10, 24);
  s.mass[MARS] = 6.39 * std::pow(10, 23);
  s.mass[JUPITER] = 1.898 * std::pow(10, 27);
  s.mass[SATURN] = 5.683 * std::pow(10, 26);
  s.mass[URANUS] = 8.681 * std::pow(10, 25);
  s.mass[NEPTUNE] = 1.024 * std::pow(10, 26);
  s.mass[MOON] = 7.342 * std::pow(10, 22);

  // Positions (in meters) and velocities (in m/s)
  double AU = 1.496 * std::pow(10, 11); // Astronomical Unit

  s.x = {0, 0.39*AU, 0.72*AU, 1.0*AU, 1.52*AU, 5.20*AU, 9.58*AU, 19.22*AU, 30.05*AU, 1.0*AU + 3.844*std::pow(10, 8)};
  s.y = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.z = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  s.vx = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  s.vy = {0, 47870, 35020, 29780, 24130, 13070, 9680, 6800, 5430, 29780 + 1022};
  s.vz = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
}

// kernel to compute force
__global__ void compute_force(size_t nbpart, double *mass, double *x, double *y, double *z, double *fx, double *fy, double *fz, double G) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const double softening = .1;
    if (idx < nbpart){
        double fx_i = 0;
        double fy_i = 0;
        double fz_i = 0;

        for(size_t j = 0; j < nbpart; ++j) {
            if (idx == j) continue;
            //direction
            double dx = x[j] - x[idx];
            double dy = y[j] - y[idx];
            double dz = z[j] - z[idx];
            double dist_sq = dx * dx + dy * dy + dz * dz + softening * softening;
            double norm = std::sqrt(dist_sq);
            double F = G * mass[idx] * mass[j] / dist_sq; //that the strength of the force
      
            //apply force
            fx_i += dx * F / norm;
            fy_i += dy * F / norm;
            fz_i += dz * F / norm;
        }
        // Stores the computed force
        fx[idx] = fx_i;
        fy[idx] = fy_i;
        fz[idx] = fz_i;
    }
}

// kernel to update velocity and position
__global__ void update_vel_pos(size_t nbpart, double *x, double *y, double *z, double *vx, double *vy, double *vz, double *fx, double *fy, double *fz, double dt, double *mass) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Thread index
    if (idx <nbpart){
        //update velocity
        vx[idx] += fx[idx] / mass[idx] * dt;
        vy[idx] += fy[idx] / mass[idx] * dt;
        vz[idx] += fz[idx] / mass[idx] * dt;

        //update position
        x[idx] += vx[idx] * dt;
        y[idx] += vy[idx] * dt;
        z[idx] += vz[idx] * dt;
    }
}

// CUDA memory allocation
void cuda_malloc(simulation& s, double** d_mass, double** d_x, double** d_y, double** d_z, double** d_vx, double** d_vy, double** d_vz, double** d_fx, double** d_fy, double** d_fz) {
    size_t size = s.nbpart * sizeof(double);
    cudaMalloc(d_mass, size);
    cudaMalloc(d_x, size);
    cudaMalloc(d_y, size);
    cudaMalloc(d_z, size);
    cudaMalloc(d_vx, size);
    cudaMalloc(d_vy, size);
    cudaMalloc(d_vz, size);
    cudaMalloc(d_fx, size);
    cudaMalloc(d_fy, size);
    cudaMalloc(d_fz, size);
}

// CUDA memory copy from host to device
void cuda_host_to_device(simulation& s, double* d_mass, double* d_x, double* d_y, double* d_z, double* d_vx, double* d_vy, double* d_vz, double* d_fx, double* d_fy, double* d_fz) {
    size_t size = s.nbpart * sizeof(double);
    cudaMemcpy(d_mass, s.mass.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, s.x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, s.y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, s.z.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, s.vx.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, s.vy.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, s.vz.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fx, s.fx.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fy, s.fy.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fz, s.fz.data(), size, cudaMemcpyHostToDevice);
}

// CUDA memory copy from device to host
void cuda_device_to_host(simulation& s, double* d_mass, double* d_x, double* d_y, double* d_z, double* d_vx, double* d_vy, double* d_vz, double* d_fx, double* d_fy, double* d_fz) {
    size_t size = s.nbpart * sizeof(double);
    cudaMemcpy(s.mass.data(), d_mass, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(s.x.data(), d_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(s.y.data(), d_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(s.z.data(), d_z, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(s.vx.data(), d_vx, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(s.vy.data(), d_vy, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(s.vz.data(), d_vz, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(s.fx.data(), d_fx, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(s.fy.data(), d_fy, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(s.fz.data(), d_fz, size, cudaMemcpyDeviceToHost);
}

// CUDA memory free
void cuda_free(double* d_mass, double* d_x, double* d_y, double* d_z, double* d_vx, double* d_vy, double* d_vz, double* d_fx, double* d_fy, double* d_fz) {
    cudaFree(d_mass);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    cudaFree(d_fx);
    cudaFree(d_fy);
    cudaFree(d_fz);
}

void dump_state(simulation& s) {
  std::cout<<s.nbpart<<'\t';
  for (size_t i=0; i<s.nbpart; ++i) {
    std::cout<<s.mass[i]<<'\t';
    std::cout<<s.x[i]<<'\t'<<s.y[i]<<'\t'<<s.z[i]<<'\t';
    std::cout<<s.vx[i]<<'\t'<<s.vy[i]<<'\t'<<s.vz[i]<<'\t';
    std::cout<<s.fx[i]<<'\t'<<s.fy[i]<<'\t'<<s.fz[i]<<'\t';
  }
  std::cout<<'\n';
}

void load_from_file(simulation& s, std::string filename) {
  std::ifstream in (filename);
  size_t nbpart;
  in>>nbpart;
  s = simulation(nbpart);
  for (size_t i=0; i<s.nbpart; ++i) {
    in>>s.mass[i];
    in >>  s.x[i] >>  s.y[i] >>  s.z[i];
    in >> s.vx[i] >> s.vy[i] >> s.vz[i];
    in >> s.fx[i] >> s.fy[i] >> s.fz[i];
  }
  if (!in.good())
    throw "kaboom";
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cerr
      <<"usage: "<<argv[0]<<" <input> <dt> <nbstep> <printevery> <cudaBlockSize>>"<<"\n"
      <<"input can be:"<<"\n"
      <<"a number (random initialization)"<<"\n"
      <<"planet (initialize with solar system)"<<"\n"
      <<"a filename (load from file in singleline tsv)"<<"\n";
    return -1;
  }
  
  double dt = std::atof(argv[2]); //in seconds
  size_t nbstep = std::atol(argv[3]);
  size_t printevery = std::atol(argv[4]);
  int cudaBlockSize = std::atoi(argv[5]); // number of threads per block
  
  simulation s(1);

  //parse command line
  {
    size_t nbpart = std::atol(argv[1]); //return 0 if not a number
    if ( nbpart > 0) {
      s = simulation(nbpart);
      random_init(s);
    } else {
      std::string inputparam = argv[1];
      if (inputparam == "planet") {
	init_solar(s);
      } else{
	load_from_file(s, inputparam);
      }
    }    
  }

  auto start = std::chrono::high_resolution_clock::now();

  double *d_mass, *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_fx, *d_fy, *d_fz;
  cuda_malloc(s, &d_mass, &d_x, &d_y, &d_z, &d_vx, &d_vy, &d_vz, &d_fx, &d_fy, &d_fz);
  cuda_host_to_device(s, d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz);

  int threadsPerBlock = cudaBlockSize;
  int blocksPerGrid = (s.nbpart + threadsPerBlock - 1) / threadsPerBlock;

  
  for (size_t step = 0; step< nbstep; step++) {
    if (step %printevery == 0){
      dump_state(s);
    }

    // Sets forces back to zero
    cudaMemset(d_fx, 0, sizeof(double)*s.nbpart);
    cudaMemset(d_fy, 0, sizeof(double)*s.nbpart);
    cudaMemset(d_fz, 0, sizeof(double)*s.nbpart);

    double Gvalue = 6.674 * std::pow(10, -11);
    // Kernel launch for compute_force
    compute_force<<<blocksPerGrid, threadsPerBlock>>>(s.nbpart, d_mass, d_x, d_y, d_z, d_fx, d_fy, d_fz, Gvalue);
    // Check for any errors after kernel launch for compute_force
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess){
      std::cerr<<"Error in compute_force: "<<cudaGetErrorString(error)<<'\n';
      return -1;
    }

    // Kernel launch for update_vel_pos
    update_vel_pos<<<blocksPerGrid, threadsPerBlock>>>(s.nbpart, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, dt, d_mass);
    // Check for any errors after kernel launch for update_vel_pos
    error = cudaGetLastError();
    if (error != cudaSuccess){
      std::cerr<<"Error in update_vel_pos: "<<cudaGetErrorString(error)<<'\n';
      return -1;
    }

    // Snychronize threads when they are finished
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess){
      std::cerr<<"Error in synchronize: "<<cudaGetErrorString(error)<<'\n';
      return -1;
    }

    // Copies data from device to host
    cuda_device_to_host(s, d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_taken = end - start;
    std::cout <<"Time taken: "<<time_taken.count() << " seconds"<<'\n';

    // Frees up the memory
  cuda_free(d_mass, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz);
  return 0;
}
