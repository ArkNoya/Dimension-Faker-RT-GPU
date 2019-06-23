/**********************************************

		Dimension Faker Raytracing GPU version

		version 0.2
		last changes	2019.6.16  10:56
		code by Ark_Noya
		email : Ark_Noya@163.com
		https://space.bilibili.com/28627837

***********************************************/

#ifndef Dimension_Faker_Raytracing_GPU_core
#define Dimension_Faker_Raytracing_GPU_core


#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#define pia 3.14159265358979323f
#define rad1 .01745329251994329f

#define uchar unsigned char
#define uint unsigned int

#define surfaceUP hi.hitPos + hi.N*1e-5f
#define surfaceDIR (hi.hitPos + ray.dir*1e-5f)

struct cu_vec2 {
	float x, y;

	__device__ __host__ cu_vec2(float i = .0) : cu_vec2(i, i) {}
	__device__ __host__ cu_vec2(float x, float y) : x(x), y(y) {}

	__device__ __host__ inline float operator[] (const int & i)const {
		return (&x)[i];
	}
	__device__ __host__ inline float & operator[] (const int & i) {
		return (&x)[i];
	}
};
__device__ __host__ inline cu_vec2 operator+(const cu_vec2 & a, const cu_vec2 & b) {
	return cu_vec2(
		a.x + b.x,
		a.y + b.y
	);
}
__device__ __host__ inline cu_vec2 operator-(const cu_vec2 & a, const cu_vec2 & b) {
	return cu_vec2(
		a.x - b.x,
		a.y - b.y
	);
}
__device__ __host__ inline cu_vec2 operator*(const cu_vec2 & a, const cu_vec2 & b) {
	return cu_vec2(
		a.x * b.x,
		a.y * b.y
	);
}
__device__ __host__ inline cu_vec2 operator/(const cu_vec2 & a, const cu_vec2 & b) {
	return cu_vec2(
		a.x / b.x,
		a.y / b.y
	);
}

__device__ __host__ inline cu_vec2 operator+=(cu_vec2 & a, const cu_vec2 & b) {
	a = a + b;
	return a;
}
__device__ __host__ inline cu_vec2 operator*=(cu_vec2 & a, const cu_vec2 & b) {
	a = a * b;
	return a;
}
__device__ __host__ inline cu_vec2 operator/=(cu_vec2 & a, const cu_vec2 & b) {
	a = a / b;
	return a;
}

__device__ __host__ inline float dot(const cu_vec2 & a, const cu_vec2 & b) {
	return a.x*b.x + a.y*b.y;
}
__device__ __host__ inline float length(const cu_vec2 & v) {
	return sqrtf(dot(v, v));
}
__device__ __host__ inline cu_vec2 normalize(const cu_vec2 & v) {
	return v / length(v);
}

struct cu_vec {
	float x, y, z;

	__device__ __host__ cu_vec(float i = .0) : cu_vec(i, i, i) {}
	__device__ __host__ cu_vec(float x, float y, float z) : x(x), y(y), z(z) {}

	__device__ __host__ inline float operator[] (const int & i)const {
		return (&x)[i];
	}
	__device__ __host__ inline float & operator[] (const int & i) {
		return (&x)[i];
	}
};
__device__ __host__ inline cu_vec operator+(const cu_vec & a, const cu_vec & b) {
	return cu_vec(
		a.x + b.x,
		a.y + b.y,
		a.z + b.z
	);
}
__device__ __host__ inline cu_vec operator-(const cu_vec & a, const cu_vec & b) {
	return cu_vec(
		a.x - b.x,
		a.y - b.y,
		a.z - b.z
	);
}
__device__ __host__ inline cu_vec operator*(const cu_vec & a, const cu_vec & b) {
	return cu_vec(
		a.x*b.x,
		a.y*b.y,
		a.z*b.z
	);
}
__device__ __host__ inline cu_vec operator/(const cu_vec & a, const cu_vec & b) {
	return cu_vec(
		a.x / b.x,
		a.y / b.y,
		a.z / b.z
	);
}
__device__ __host__ inline cu_vec operator+=(cu_vec & a, const cu_vec & b) {
	a = a + b;
	return a;
}
__device__ __host__ inline cu_vec operator*=(cu_vec & a, const cu_vec & b) {
	a = a * b;
	return a;
}
__device__ __host__ inline cu_vec operator/=(cu_vec & a, const cu_vec & b) {
	a = a / b;
	return a;
}

__device__ __host__ inline cu_vec cross(const cu_vec & a, const cu_vec & b) {
	return cu_vec(
		a.y*b.z - a.z*b.y,
		a.z*b.x - a.x*b.z,
		a.x*b.y - a.y*b.x
	);
}
__device__ __host__ inline float dot(const cu_vec & a, const cu_vec & b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ __host__ inline float length(const cu_vec & v) {
	return sqrtf(dot(v, v));
}
__device__ __host__ inline cu_vec normalize(const cu_vec & v) {
	return v / length(v);
}
__device__ __host__ inline float angle(const cu_vec & a, const cu_vec & b) {
	return acosf(dot(normalize(a), normalize(b)));
}

__device__ inline cu_vec cu_randvec(curandState * randStates) {
	return normalize(cu_vec(
		curand_uniform(randStates),
		curand_uniform(randStates),
		curand_uniform(randStates)
	) * 2 - 1);
}
__device__ inline cu_vec2 cu_randvec2(curandState * randStates) {
	return normalize(cu_vec2(
		curand_uniform(randStates),
		curand_uniform(randStates)
	) * 2 - 1);
}
__global__ void rand_init(curandState * randStates, int w, int h) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= w || y >= h)return;
	int i = x + y * w;

	curand_init(2008, i, 0, &randStates[i]);
}

struct cu_Ray {
	cu_vec org, dir;

	__device__ __host__ cu_Ray(cu_vec org, cu_vec dir)
		:org(org), dir(normalize(dir)) {}

	__device__ __host__ inline cu_vec rich(const float & t)const {
		return org + dir * t;
	}
};

struct cu_hitInfo;
struct cu_scatterInfo {
	cu_Ray outRay;
	cu_vec mult;

	__device__ __host__ cu_scatterInfo(cu_Ray outRay, cu_vec mult)
		: outRay(outRay), mult(mult) {}
};
struct cu_material {
	__device__ __host__ virtual cu_scatterInfo scatter(const cu_hitInfo & hi, const cu_Ray & ray,
		curandState * randStates)const = 0;
};
struct cu_hitInfo {
	bool isHit;
	double t;
	cu_vec hitPos, N;
	cu_material * mate;

	__device__ __host__ cu_hitInfo(bool isHit, double t, cu_vec hitPos, cu_vec N, cu_material * mate)
		:isHit(isHit), t(t), hitPos(hitPos), N(normalize(N)), mate(mate) {}
};
struct cu_texture {
	__device__ __host__ virtual cu_vec value(const cu_hitInfo & hi, const cu_Ray & ray)const = 0;
};
struct flat : cu_texture {
	cu_vec color;

	__device__ __host__ flat(cu_vec color = cu_vec(.8)) : color(color) {}

	__device__ __host__ cu_vec value(const cu_hitInfo & hi, const cu_Ray & ray)const {
		return color;
	}
};
struct checkBoard : cu_texture {
	cu_vec color;
	float sx, sz, minMult;

	__device__ __host__ checkBoard(cu_vec color = cu_vec(.8),
		float sx = 4, float sz = 4, float minMult = .09)
		: color(color), sx(sx), sz(sz), minMult(minMult) {}

	__device__ __host__ cu_vec value(const cu_hitInfo & hi, const cu_Ray & ray)const {
		float mult0 = sinf(hi.hitPos.x*sx) > 0 ? 1 : 0;
		float mult1 = cosf(hi.hitPos.z*sz) > 0 ? 1 : 0;

		float mult0a = sinf(hi.hitPos.x*sx) > 0 ? 0 : 1;
		float mult1a = cosf(hi.hitPos.z*sz) > 0 ? 0 : 1;

		float allm = mult0 * mult1;
		float allma = mult0a * mult1a;

		float allmaa = allm + allma;

		if (allmaa == 0) allmaa = minMult;

		return color * allmaa;
	}
};
struct toonEdge : cu_texture {
	cu_vec color;
	float size;

	__device__ __host__ toonEdge(cu_vec color = cu_vec(.8), float size = 18)
		: color(color), size(size) {}

	__device__ __host__ cu_vec value(const cu_hitInfo & hi, const cu_Ray & ray)const {
		float a = angle(ray.dir, hi.N*-1);

		if (a > (90 - size)*rad1)
			return 0;

		return color;
	}
};
struct diffuse : cu_material {
	cu_texture * color;

	__device__ __host__ diffuse(cu_texture * color) : color(color) {}

	__device__ __host__ cu_scatterInfo scatter(const cu_hitInfo & hi, const cu_Ray & ray,
		curandState * randStates)const {

		cu_vec ov = cu_randvec(randStates) + hi.N;
		cu_vec mult = dot(normalize(ov), hi.N);

		cu_vec c = color->value(hi, ray);

		return cu_scatterInfo(cu_Ray(surfaceUP, ov), c*mult);
	}
};
__device__ __host__ inline cu_vec reflaction(const cu_vec & inRay, const cu_vec & N,
	const float & roughness, curandState * randStates) {

	cu_vec ov = inRay + N * 2 * abs(dot(inRay, N));

	if (roughness > 0) {
		cu_vec rv;
		do {
			rv = cu_randvec(randStates)*roughness;
			ov += rv;
			ov = normalize(ov);
		} while (angle(ov, N) > pia / 2);
	}

	return normalize(ov);
}
struct metal : cu_material {
	cu_vec color;
	float roughness;

	__device__ __host__ metal(cu_vec color = cu_vec(.8), float roughness = .0)
		: color(color), roughness(roughness) {}

	__device__ __host__ cu_scatterInfo scatter(const cu_hitInfo & hi, const cu_Ray & ray,
		curandState * randStates)const {

		cu_vec ov = reflaction(ray.dir, hi.N, roughness, randStates);

		return cu_scatterInfo(cu_Ray(surfaceUP, ov), color);
	}
};
__device__ __host__ inline cu_vec refraction(const cu_vec & inRay, const cu_vec & N,
	const float & ior, const float & roughness, curandState * randStates, int & crossDIR) {

	bool in = false;
	if (angle(N, inRay) < pia / 2)
		in = true;

	if (in) {
		float ia = angle(N, inRay);
		float sa = asinf(1 / ior);

		if (ia > sa) {
			crossDIR = -1;
			return reflaction(inRay, N, roughness, randStates);
		}

		float oa = asinf(sinf(ia)*ior);

		cu_vec dir = normalize(cross(N, cross(inRay, N)));

		cu_vec ov = dir * sinf(oa) + N * cosf(oa);

		if (roughness > 0) {
			cu_vec rv;
			do {
				rv = cu_randvec(randStates)*roughness;
				ov = normalize(ov + rv);
			} while (angle(ov, N) > pia / 2);
		}

		return ov;
	}
	else {
		float ia = angle(N*-1, inRay);
		float oa = asinf(sinf(ia) / ior);

		cu_vec dir = normalize(cross(N, cross(inRay, N)));

		cu_vec ov = dir * sinf(oa) + N * -1 * cosf(oa);

		if (roughness > 0) {
			cu_vec rv;
			do {
				rv = cu_randvec(randStates)*roughness;
				ov = normalize(ov + rv);
			} while (angle(ov, N*-1) > pia / 2);
		}

		return ov;
	}
}
struct snell : cu_material {
	cu_vec color;
	float ior;
	float roughness;

	__device__ __host__ snell(cu_vec color = cu_vec(.8), float ior = 1.33, float roughness = 0)
		: color(color), ior(ior), roughness(roughness) {}

	__device__ __host__ cu_scatterInfo scatter(const cu_hitInfo & hi, const cu_Ray & ray,
		curandState * randStates)const {

		int dir = 1;

		cu_vec ov = refraction(ray.dir, hi.N, ior, roughness, randStates, dir);

		return cu_scatterInfo(cu_Ray(surfaceDIR*dir, ov), color);
	}
};

struct cu_obj {
	__device__ __host__ virtual cu_hitInfo hit(const cu_Ray & ray)const = 0;
};
struct cu_Sphere : cu_obj {
	cu_vec cen;
	float rad;
	cu_material * mate;

	__device__ __host__ cu_Sphere(cu_vec cen, double rad, cu_material * mate)
		:cen(cen), rad(rad), mate(mate) {}

	__device__ __host__ cu_hitInfo hit(const cu_Ray & ray)const {
		cu_vec oc = ray.org - cen;
		float b = 2.*dot(oc, ray.dir);
		float c = dot(oc, oc) - rad * rad;
		float delta = b * b - 4.*c;

		if (delta < 0)
			return cu_hitInfo(false, INFINITY, ray.org, cu_vec(0), mate);

		float t = (-b - sqrt(delta)) / 2;

		if (length(oc) > rad && t < 0)
			return cu_hitInfo(false, INFINITY, ray.org, cu_vec(0), mate);

		if (length(oc) < rad)
			t = (-b + sqrt(delta)) / 2;

		cu_vec hp = ray.rich(t);

		return cu_hitInfo(true, t, hp, hp - cen, mate);
	}
};
struct cu_obj_list : cu_obj {
	cu_obj ** list;
	int list_size;

	__device__ __host__ cu_obj_list(cu_obj ** list, int list_size)
		:list(list), list_size(list_size) {}

	__device__ __host__ cu_hitInfo hit(const cu_Ray & ray)const {
		cu_hitInfo fhi = list[0]->hit(ray);

		for (int i = 0; i < list_size - 1; i++) {
			cu_hitInfo chi = list[i + 1]->hit(ray);
			if (chi.t < fhi.t)
				fhi = chi;
		}

		return fhi;
	}
};

struct cu_Camera {
	cu_vec org, lookAt, up, dir, rdir, sup;
	float fov;
	float fsize;

	__device__ __host__ cu_Camera(cu_vec org = cu_vec(5), cu_vec lookAt = cu_vec(0),
		float fov = 50, float fsize = .05, cu_vec up = cu_vec(0, 1, 0))
		:org(org), lookAt(lookAt), fov(fov), up(up), fsize(fsize) {
		dir = lookAt - org;
		rdir = normalize(cross(dir, up));
		sup = normalize(cross(rdir, dir));
	}

	__device__ __host__ cu_Ray camRay(const int & w, const int & h, const cu_vec2 & cp, curandState * randStates)const {
		cu_vec npc = org + normalize(dir);
		float ratio = w * 1. / h;
		float rh = atanf(fov / 2 * rad1);
		float rw = rh * ratio;

		float ox = cp.x*1. / w * 2 - 1;
		float oy = cp.y*1. / h * 2 - 1;

		cu_vec ncp = npc + rdir * rw*ox + sup * rh*oy;
		cu_vec totarv = normalize(ncp - org);

		if (fsize > 0) {

			float tardis = length(dir) - 1;

			cu_vec2 rv2 = cu_randvec2(randStates)*fsize;
			cu_vec nrcp = npc + rdir * rw*ox*rv2.x + sup * rh*oy*rv2.y;

			cu_vec tar = ncp + totarv * tardis;

			cu_vec nv = normalize(tar - nrcp);

			return cu_Ray(nrcp - nv, nv);
		}

		return cu_Ray(org, totarv);
	}
};


struct BMHead {
protected:
	char BM[2] = { 0x42,0x4d };
	char fileSize[4];
	char keepArea0[2] = { 0x00,0x00 };
	char keepArea1[2] = { 0x00,0x00 };
	char imageDataStart[4] = { 0x36,0x00,0x00,0x00 };
	char headFileSize[4] = { 0x28,0x00,0x00,0x00 };
	char swidth[4];
	char sheight[4];
	char deviceLevel[2] = { 0x01,0x00 };
	char scolorBit[2] = { 0x18,0x00 };
	char Nul[24];
public:
	BMHead(int width, int height) {
		int pn = width * height * 3 + 54;
		for (int i = 0; i < 4; i++) {
			fileSize[i] = pn % 256;
			pn /= 256;
		}
		for (int i = 0; i < 4; i++) {
			swidth[i] = width % 256;
			width /= 256;
		}
		for (int i = 0; i < 4; i++) {
			sheight[i] = height % 256;
			height /= 256;
		}
		for (int i = 0; i < 24; i++) {
			Nul[i] = 0x00;
		}
	}
	int size() {
		return 54;
	}
	char operator[](int i) const {
		return BM[i];
	}
};
ofstream & BMofs(ofstream & ofs, cu_vec v) {
	ofs << (uchar)((int)round(v.z)) << (uchar)((int)round(v.y)) << (uchar)((int)round(v.x));
	return ofs;
}
ofstream & BMofs(ofstream & ofs, int r, int g, int b) {
	ofs << (uchar)b << (uchar)g << (uchar)r;
	return ofs;
}

ostream & operator<<(ostream & os, const cu_vec & v) {
	os << "(" << v.x << "," << v.y << "," << v.z << ")";
	return os;
}

#endif // !Dimension_Faker_Raytracing_GPU_core
