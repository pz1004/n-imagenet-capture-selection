// Exact derivatives of the three calibration losses, evaluated over all classes.
// Rows store log p_vertical, log p_horizontal and their elementwise minimum.
#include <cmath>
#include <cstring>

extern "C" void row_derivatives(const double* features, int n, int k, const int* labels,
                                const int* counts, const double* theta, double* derivatives,
                                bool aggregate, bool hessian) {
    double e_stack[1000], e_first[1000], e_second[1000];
    if (aggregate) std::memset(derivatives, 0, 13*sizeof(double));
    for (int i = 0; i < n; ++i) {
        if (counts[i] == 0) continue;
        const double* a = features + (size_t)i*3*k;
        const double* b = a + k;
        const double* m = b + k;
        #pragma omp simd
        for (int j = 0; j < k; ++j) {
            e_stack[j] = std::exp(theta[0]*a[j] + theta[1]*b[j] + theta[2]*m[j]);
            e_first[j] = std::exp(theta[3]*a[j]);
            e_second[j] = std::exp(theta[4]*b[j]);
        }
        double s=0, sa=0, sb=0, sm=0, saa=0, sab=0, sam=0, sbb=0, sbm=0, smm=0;
        double u=0, ua=0, uaa=0, v=0, vb=0, vbb=0;
        #pragma omp simd reduction(+:s,sa,sb,sm,u,ua,v,vb)
        for (int j = 0; j < k; ++j) {
            s += e_stack[j]; sa += e_stack[j]*a[j]; sb += e_stack[j]*b[j];
            sm += e_stack[j]*m[j]; u += e_first[j]; ua += e_first[j]*a[j];
            v += e_second[j]; vb += e_second[j]*b[j];
        }
        const double ma = sa/s, mb = sb/s, mm = sm/s, mu=ua/u, mv=vb/v;
        const int y=labels[i];
        double row[13] = {ma-a[y],mb-b[y],mm-m[y],mu-a[y],mv-b[y]};
        if (hessian) {
            #pragma omp simd reduction(+:saa,sab,sam,sbb,sbm,smm,uaa,vbb)
            for (int j = 0; j < k; ++j) {
                // Center once to avoid subtractive cancellation in covariances.
                const double da=a[j]-ma, db=b[j]-mb, dm=m[j]-mm;
                saa+=e_stack[j]*da*da; sab+=e_stack[j]*da*db; sam+=e_stack[j]*da*dm;
                sbb+=e_stack[j]*db*db; sbm+=e_stack[j]*db*dm; smm+=e_stack[j]*dm*dm;
                uaa+=e_first[j]*(a[j]-mu)*(a[j]-mu);
                vbb+=e_second[j]*(b[j]-mv)*(b[j]-mv);
            }
            row[5]=saa/s; row[6]=sab/s; row[7]=sam/s; row[8]=sbb/s;
            row[9]=sbm/s; row[10]=smm/s; row[11]=uaa/u; row[12]=vbb/v;
        }
        if (aggregate) {
            for (int j=0; j<13; ++j) derivatives[j] += counts[i]*row[j];
        } else {
            std::memcpy(derivatives+(size_t)i*13, row, 13*sizeof(double));
        }
    }
}
