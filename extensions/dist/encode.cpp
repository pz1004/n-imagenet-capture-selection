// DiST on the upstream 224 x 224 grid, using the same 5 x 5 temporal discount
// and normalized rank among unique discounted timestamps as the released parser.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <utility>

constexpr int width = 224;
constexpr int pixels = width * width;

extern "C" int encode_dist(const int* position, const double* times, int n, float* output) {
    std::fill(output, output + 2*pixels, 0.f);
    if (n == 0) return 0;  // Placeholder only, never used for incomplete second requests.
    double span = times[n-1] - times[0];
    if (span <= 0) return 1;
    for (int channel=0; channel<2; ++channel) {
        std::vector<int> counts(pixels, 0);
        std::vector<float> first(pixels, 1.f), last(pixels, 0.f);
        int largest = 0;
        for (int j=0; j<n; ++j) {
            int i = position[j] - pixels*channel;
            if (i<0 || i>=pixels) continue;
            float t=static_cast<float>((times[j]-times[0])/span);
            largest=std::max(largest, ++counts[i]);
            first[i]=std::min(first[i],t);
            last[i]=std::max(last[i],t);
        }
        std::vector<int> histogram(largest+1);
        for (int k: counts) ++histogram[k];
        int clip=0, cumulative=0;
        for(int frequency: histogram) {
            if (!frequency) continue;
            cumulative+=frequency;
            if (cumulative < pixels*.99) ++clip;
        }
        for (int i=0; i<pixels; ++i) {
            counts[i]=std::min(clip,counts[i]);
            if (!counts[i]) first[i]=1.f;
        }
        std::vector<std::pair<float,int>> positive;
        positive.reserve(n);
        for (int i=0; i<pixels; ++i) {
            if (last[i]<=0) continue;
            const int x=i%width, y=i/width;
            int count=0;
            float hi=0.f, lo=1.f;
            for (int yy=std::max(0,y-2);yy<=std::min(width-1,y+2);++yy) {
                for (int xx=std::max(0,x-2);xx<=std::min(width-1,x+2);++xx) {
                    int k=yy*width+xx;
                    count+=counts[k];
                    hi=std::max(hi,last[k]);
                    lo=std::min(lo,first[k]);
                }
            }
            // Match torch's float avg_pool2d and subsequent multiply.
            float pooled=(static_cast<float>(count)/25.f)*25.f;
            float value=last[i];
            if (counts[i]) value-=3.f*((hi-lo)/pooled);
            if (pooled!=1.f && value>0.f) positive.emplace_back(value,i);
        }
        std::sort(positive.begin(),positive.end());
        int rank=0;
        float previous=0.f;
        for (auto p: positive) {
            if (p.first!=previous) ++rank;
            previous=p.first;
            output[channel*pixels+p.second]=static_cast<float>(rank);
        }
        float size=static_cast<float>(rank+1);
        for(auto p:positive) output[channel*pixels+p.second]/=size;
    }
    return 0;
}
