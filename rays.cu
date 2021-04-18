#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "mpi.h"

#define FATAL(description)                                      \
    do {                                                        \
        std::cerr << "Error in " << __FILE__ << ":" << __LINE__ \
                  << ". Message: " << description << std::endl; \
        exit(0);                                                \
    } while (0)

#define CHECK_CUDART(call)                  \
    do {                                    \
        cudaError_t res = call;             \
        if (res != cudaSuccess) {           \
            FATAL(cudaGetErrorString(res)); \
        }                                   \
    } while (0)

#define CHECK_MPI(call)                        \
    do {                                       \
        int res = call;                        \
        if (res != MPI_SUCCESS) {              \
            char desc[MPI_MAX_ERROR_STRING];   \
            int len;                           \
            MPI_Error_string(res, desc, &len); \
            MPI_Finalize();                    \
            FATAL(desc);                       \
        }                                      \
    } while (0)

void handle_signals() {
    auto nop_handler = [](int sig) {
        std::cerr << "Unexpected signal received: " << sig << std::endl;
        exit(0);
    };
    std::signal(SIGSEGV, nop_handler);
    std::signal(SIGABRT, nop_handler);
}

struct MPIContext {
    MPIContext(int *argc, char ***argv) { CHECK_MPI(MPI_Init(argc, argv)); }
    ~MPIContext() {
        CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
        CHECK_MPI(MPI_Finalize());
    }
};

template <typename T>
struct Vector3 {
    T x, y, z;

    friend std::istream &operator>>(std::istream &is, Vector3 &v) {
        is >> v.x >> v.y >> v.z;
        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, const Vector3 &v) {
        os << '[' << v.x << ' ' << v.y << ' ' << v.z << ']';
        return os;
    }
};

using Vector3d = Vector3<double>;

struct CylindricalMovementParams {
    double r0, z0, phi0, ar, az, wr, wz, wphi, pr, pz;

    friend std::istream &operator>>(std::istream &is,
                                    CylindricalMovementParams &p) {
        is >> p.r0 >> p.z0 >> p.phi0 >> p.ar >> p.az >> p.wr >> p.wz >>
            p.wphi >> p.pr >> p.pz;
        return is;
    }

    friend std::ostream &operator<<(std::ostream &os,
                                    const CylindricalMovementParams &p) {
        os << p.r0 << ' ' << p.z0 << ' ' << p.phi0 << ' ' << p.ar << ' ' << p.az
           << ' ' << p.wr << ' ' << p.wz << ' ' << p.wphi << ' ' << p.pr << ' '
           << p.pz;
        return os;
    }
};

struct FigureParams {
    Vector3d center;
    Vector3d color;
    double radius;
    double kreflection, krefraction;
    int nlights;

    friend std::istream &operator>>(std::istream &is, FigureParams &p) {
        is >> p.center >> p.color >> p.radius >> p.kreflection >>
            p.krefraction >> p.nlights;
        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, const FigureParams &p) {
        os << p.center << ' ' << p.color << ' ' << p.radius << ' '
           << p.kreflection << ' ' << p.krefraction << ' ' << p.nlights;
        return os;
    }
};

struct FloorParams {
    Vector3d a, b, c, d;
    std::string texture_path;
    Vector3d color;
    double kreflection;

    friend std::istream &operator>>(std::istream &is, FloorParams &p) {
        is >> p.a >> p.b >> p.c >> p.d >> p.texture_path >> p.color >>
            p.kreflection;
        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, const FloorParams &p) {
        os << p.a << ' ' << p.b << ' ' << p.c << ' ' << p.d << ' '
           << p.texture_path << ' ' << p.color << ' ' << p.kreflection;
        return os;
    }
};

struct LightParams {
    Vector3d pos;
    Vector3d color;

    friend std::istream &operator>>(std::istream &is, LightParams &p) {
        is >> p.pos >> p.color;
        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, const LightParams &p) {
        os << p.pos << ' ' << p.color;
        return os;
    }
};

struct Params {
    int nframes;
    std::string output_pattern;
    int w, h;
    double angle;
    CylindricalMovementParams camera_center, camera_dir;
    FigureParams hex, octa, icos;
    FloorParams floor;
    int nlights;
    std::vector<LightParams> lights;

    friend std::istream &operator>>(std::istream &is, Params &p) {
        is >> p.nframes >> p.output_pattern >> p.w >> p.h >> p.angle >>
            p.camera_center >> p.camera_dir >> p.hex >> p.octa >> p.icos >>
            p.floor >> p.nlights;
        p.lights.resize(p.nlights);
        for (auto &it : p.lights) is >> it;
        return is;
    }

    friend std::ostream &operator<<(std::ostream &os, const Params &p) {
        os << p.nframes << ' ' << p.output_pattern << '\n'
           << p.w << ' ' << p.h << ' ' << p.angle << '\n'
           << "camera center: " << p.camera_center << '\n'
           << "camera dir: " << p.camera_dir << '\n'
           << "hex: " << p.hex << '\n'
           << "octa: " << p.octa << '\n'
           << "icos: " << p.icos << '\n'
           << "floor: " << p.floor << '\n'
           << "nlights: " << p.nlights << '\n';
        for (auto &it : p.lights) os << it;
        return os;
    }
};

struct Trig {
    Vector3d a;
    Vector3d b;
    Vector3d c;
    Vector3d color;
};

static std::vector<Trig> scene_trigs;

std::vector<std::string> split_string(const std::string &s, char d) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string word;
    while (getline(ss, word, d)) {
        result.push_back(word);
    }
    return result;
}

double dot_product(Vector3d a, Vector3d b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector3d cross_product(Vector3d a, Vector3d b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

double norm(Vector3d v) { return sqrt(dot_product(v, v)); }

Vector3d normalize(Vector3d v) {
    double l = norm(v);
    return {v.x / l, v.y / l, v.z / l};
}

Vector3d diff(Vector3d a, Vector3d b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vector3d add(Vector3d a, Vector3d b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vector3d mult(Vector3d a, Vector3d b, Vector3d c, Vector3d v) {
    return {a.x * v.x + b.x * v.y + c.x * v.z,
            a.y * v.x + b.y * v.y + c.y * v.z,
            a.z * v.x + b.z * v.y + c.z * v.z};
}

Vector3d mult(Vector3d a, double k) { return {k * a.x, k * a.y, k * a.z}; }

Vector3d div(Vector3d a, double k) { return {a.x / k, a.y / k, a.z / k}; }

uchar4 color_from_normalized(Vector3d v) {
    return make_uchar4(255. * v.x, 255. * v.y, 255. * v.z, 0u);
}

void import_obj_file(const std::string &filepath, const FigureParams &fp) {
    std::ifstream is(filepath);
    if (!is) {
        std::string desc = "can't open " + filepath;
        FATAL(desc);
    }

    double r = 0;
    std::vector<Vector3d> vertices;
    std::vector<Trig> figure_trigs;
    std::string line;
    while (std::getline(is, line)) {
        std::vector<std::string> buffer = split_string(line, ' ');
        if (line.empty()) {
            continue;
        } else if (buffer[0] == "v") {
            double x = std::stod(buffer[2]);
            double y = std::stod(buffer[3]);
            double z = std::stod(buffer[4]);

            vertices.push_back({x, y, z});
        } else if (buffer[0] == "f") {
            std::vector<std::string> indexes = split_string(buffer[1], '/');
            Vector3d a = vertices[std::stoi(indexes[0]) - 1];
            indexes = split_string(buffer[2], '/');
            Vector3d b = vertices[std::stoi(indexes[0]) - 1];
            indexes = split_string(buffer[3], '/');
            Vector3d c = vertices[std::stoi(indexes[0]) - 1];

            r = std::max(r, norm(a));
            r = std::max(r, norm(b));
            r = std::max(r, norm(c));

            figure_trigs.push_back(Trig{a, b, c, fp.color});
        }
    }
    for (auto &it : figure_trigs) {
        double k = fp.radius / r;
        Vector3d a = add(mult(it.a, k), fp.center);
        Vector3d b = add(mult(it.b, k), fp.center);
        Vector3d c = add(mult(it.c, k), fp.center);
        scene_trigs.push_back({a, b, c, it.color});
    }
}

void add_floor(const FloorParams &fp) {
    scene_trigs.push_back({fp.a, fp.b, fp.c, fp.color});
    scene_trigs.push_back({fp.a, fp.d, fp.c, fp.color});
}

uchar4 ray(Vector3d pos, Vector3d dir) {
    int k, k_min = -1;
    double ts_min;
    for (k = 0; k < scene_trigs.size(); k++) {
        Vector3d e1 = diff(scene_trigs[k].b, scene_trigs[k].a);
        Vector3d e2 = diff(scene_trigs[k].c, scene_trigs[k].a);
        Vector3d p = cross_product(dir, e2);
        double div = dot_product(p, e1);
        if (fabs(div) < 1e-10) continue;
        Vector3d t = diff(pos, scene_trigs[k].a);
        double u = dot_product(p, t) / div;
        if (u < 0.0 || u > 1.0) continue;
        Vector3d q = cross_product(t, e1);
        double v = dot_product(q, dir) / div;
        if (v < 0.0 || v + u > 1.0) continue;
        double ts = dot_product(q, e2) / div;
        if (ts < 0.0) continue;
        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
        }
    }
    if (k_min == -1) return {0, 0, 0, 0};
    return color_from_normalized(scene_trigs[k_min].color);
}

void render(Vector3d pc, Vector3d pv, int w, int h, double angle,
            uchar4 *data) {
    int i, j;
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    Vector3d bz = normalize(diff(pv, pc));
    Vector3d bx = normalize(cross_product(bz, {0.0, 0.0, 1.0}));
    Vector3d by = normalize(cross_product(bx, bz));
    for (i = 0; i < w; i++)
        for (j = 0; j < h; j++) {
            Vector3d v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
            Vector3d dir = mult(bx, by, bz, v);
            data[(h - 1 - j) * w + i] = ray(pc, normalize(dir));
        }
}

int main(int argc, char *argv[]) {
    handle_signals();
    // MPIContext ctx(&argc, &argv);

    Params params;
    std::cin >> params;
    std::cerr << params << std::endl;

    import_obj_file("hex.obj", params.hex);
    add_floor(params.floor);

    int w = 640, h = 480;
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    Vector3d pc, pv;

    // build_space();

    for (int frame = 0; frame < params.nframes; ++frame) {
        pc = (Vector3d){6.0 * sin(0.05 * frame), 6.0 * cos(0.05 * frame),
                        5.0 + 2.0 * sin(0.1 * frame)};
        pv = (Vector3d){3.0 * sin(0.05 * frame + M_PI),
                        3.0 * cos(0.05 * frame + M_PI), 0.0};
        render(pc, pv, w, h, 120.0, data);

        char output_path[256];
        sprintf(output_path, params.output_pattern.data(), frame);
        printf("%d: %s\n", frame, output_path);

        FILE *out = fopen(output_path, "wb");
        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(data, sizeof(uchar4), w * h, out);
        fclose(out);
    }
    free(data);
    return 0;
}