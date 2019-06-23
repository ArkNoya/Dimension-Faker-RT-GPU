/*******************************************

	Dimension Faker Raytracing GPU version

	Ark_Noya@163.com
	https://space.bilibili.com/28627837

	last changes 2019.6.23

********************************************/


#include "core.h"
#include <time.h>
#include <vector>

#include <GLFW/glfw3.h>

__device__ __host__ cu_vec gamma(const cu_vec & c) {
	return cu_vec(
		sqrtf(c.x),
		sqrtf(c.y),
		sqrtf(c.z)
	);
}
__device__ __host__ cu_vec atangamma(const cu_vec & c) {
	return cu_vec(
		atanf(c.x*pia / 2),
		atanf(c.y*pia / 2),
		atanf(c.z*pia / 2)
	);
}

__device__ __host__ float clamp01(const float & i) {
	if (i > 1)
		return 1;
	if (i < 0)
		return 0;

	return i;
}

__device__ __host__ cu_vec clamp01(const cu_vec & v) {
	return cu_vec(
		clamp01(v.x),
		clamp01(v.y),
		clamp01(v.z)
	);
}

__global__ void build_scenes(cu_obj ** scenes, cu_obj ** list) {
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		*list = new cu_Sphere(cu_vec(0, 1.31, 0), 1.3, new diffuse(new toonEdge(cu_vec(.99, .6, .5))));
		*(list + 1) = new cu_Sphere(cu_vec(0, -200.01, 0), 200, new diffuse(new checkBoard));
		*(list + 2) = new cu_Sphere(cu_vec(-3.7, 2.51, 0), 2.5, new metal(cu_vec(.99, .6, .55), .02));
		*(list + 3) = new cu_Sphere(cu_vec(3.7, 2.51, 0), 2.5, new metal(cu_vec(.95, .99, .7), .1));
		*(list + 4) = new cu_Sphere(cu_vec(-1.95, 2.5, 6), 1.2, new diffuse(new flat(cu_vec(.6, .99, .65))));
		*(list + 5) = new cu_Sphere(cu_vec(-.575, .9, 2.91), .8, new snell(cu_vec(.7, .8, .99), 1.05, .002));
		*scenes = new cu_obj_list(list, 6);
	}
}
__global__ void free_scenes(cu_obj ** scenes, cu_obj ** list) {
	delete *list;
	delete *(list + 1);
	delete *(list + 2);
	delete *(list + 3);
	delete *(list + 4);
	delete *(list + 5);
	delete *scenes;
}

__global__ void GPU_shading(cu_vec * fb, cu_obj ** scenes, cu_Camera * cam, int w, int h, curandState * randStates, int sample_times = 6, int depth = 8) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= w || y >= h)return;
	int pdp = x + y * w;

	cu_vec2 cp(x, y);

	curandState locRS = randStates[pdp];

	for (int i = 0; i < sample_times; i++) {
		cp.x += (curand_uniform(&locRS) * 2 - 1)*.15;
		cp.y += (curand_uniform(&locRS) * 2 - 1)*.15;
		cu_Ray cr = cam->camRay(w, h, cp, randStates);

		cu_vec c;
		cu_vec allmult = cu_vec(1);

		for (int j = 0; j < depth; j++) {
			cu_hitInfo hi = (*scenes)->hit(cr);

			if (hi.isHit) {
				cu_scatterInfo si = hi.mate->scatter(hi, cr, &locRS);
				cr = si.outRay;
				cu_vec mult = si.mult;
				allmult *= mult;
			}
			else {
				float lum = angle(cu_vec(0, 1, 0), cr.dir) / pia / 2;
				c = cu_vec(1)*lum + cu_vec(.55, .65, .99)*(1 - lum);
				break;
			}

		}

		fb[pdp] += c * allmult;
	}
	fb[pdp] /= sample_times;

}

int w = 1024;
int h = 768;
bool changed = true;
bool first = true;

void dynamicResizeWindow(GLFWwindow * window, int iw, int ih) {
	w = iw;
	h = ih;
	first = true;
	changed = true;
	glViewport(0, 0, w, h);
}

int main() {

	//create scenes
	cu_obj ** list;
	cudaMallocManaged((void**)&list, sizeof(cu_obj*) * 6);
	cu_obj ** scenes;
	cudaMallocManaged((void**)&scenes, sizeof(cu_obj*));
	build_scenes << <1, 1 >> > (scenes, list);
	cudaDeviceSynchronize();
	//create camera
	cu_Camera * cam0;
	cudaMallocManaged((void**)&cam0, sizeof(cu_Camera));
	*cam0 = cu_Camera(cu_vec(.25, 2.8, 7), cu_vec(0, .6, 0), 66, .06);

	//result window
	glfwInit();
	GLFWwindow * window = glfwCreateWindow(w, h, "Dimension Faker Raytracing GPU result", NULL, NULL);
	glfwMakeContextCurrent(window);

	vector<cu_vec> colors(w*h);

	while (!glfwWindowShouldClose(window)) {
		glfwSetFramebufferSizeCallback(window, dynamicResizeWindow);

		if (first) {
			glClearColor(.05, .1, .1, 1.);
			glClear(GL_COLOR_BUFFER_BIT);
			glfwSwapBuffers(window);

			first = false;
		}

		if (changed) {

			// create frame buffer
			cu_vec * fb;
			cudaMallocManaged((void**)&fb, sizeof(cu_vec)*w*h);

			//create curandStates
			curandState * d_randStates;
			cudaMallocManaged((void**)&d_randStates, sizeof(curandState)*w*h);

			//set threads Grid
			int block_length = 16;
			dim3 Grid(w / block_length + 1, h / block_length + 1);
			dim3 Block(block_length, block_length);
			//initial curandStates
			int t0 = time(NULL);
			rand_init << <Grid, Block >> > (d_randStates, w, h);
			cudaDeviceSynchronize();
			int t1 = time(NULL);
			cout << "CUDA Random States Initial OK ! time used : " << t1 - t0 << " second !" << endl;
			//rendering
			GPU_shading << <Grid, Block >> > (fb, scenes, cam0, w, h, d_randStates, 64, 16);
			cudaDeviceSynchronize();
			int t2 = time(NULL);
			cout << "GPU Rendering OK ! time used : " << t2 - t1 << " second !" << endl;

			//free curandStates
			cudaFree(d_randStates);

			for (int i = 0; i < w*h; i++) {

				float x = i % w;
				float y = i / w;

				cu_vec c = atangamma(clamp01(fb[i]));

				glBegin(GL_POINTS);

				glVertex2f(x / w * 2 - 1, y / h * 2 - 1);
				glColor3f(c.x, c.y, c.z);

				glEnd();

			}
			glfwSwapBuffers(window);

			colors.resize(w*h);

			for (int j = 0; j < w*h; j++) {
				colors[j] = fb[j];
			}

			//free frame buffer
			cudaFree(fb);

			changed = false;
		}

		glfwPollEvents();
	}
	glfwTerminate();

	//free scenes
	free_scenes << <1, 1 >> > (scenes, list);
	cudaDeviceSynchronize();
	cudaFree(cam0);

	//create bmp out file stream
	ofstream ofs("result.bmp", ios::out | ios::binary);
	BMHead bmh(w, h);
	for (int i = 0; i < 54; i++)ofs << bmh[i];

	//write file
	for (int i = 0; i < w*h; i++) {
		BMofs(ofs, atangamma(clamp01(colors[i])) * 255);
	}

	//open result
	ofs.close();
	system("result.bmp");

	return 0;
}