#include <immintrin.h>
#include <stdint.h>
#include<math.h>
#ifdef __GNUC__
    #define ALIGN_ATTRIBUTE(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
    #define ALIGN_ATTRIBUTE(n) __declspec(align(n))
#else
    #error "Unsupported compiler"
#endif

#define PORTABLE_ALIGN32 ALIGN_ATTRIBUTE(32)


#ifndef _compute2_
#define _compute2_
static float
CosineDistanceAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    size_t qty8 = qty >> 3;

    const float *pEnd1 = pVect1 + (qty8 << 3);

    __m256 dot_product_sum = _mm256_setzero_ps();
    __m256 squared_sum1 = _mm256_setzero_ps();
    __m256 squared_sum2 = _mm256_setzero_ps();

    while (pVect1 < pEnd1) {
        __m256 v1 = _mm256_loadu_ps(pVect1);
        __m256 v2 = _mm256_loadu_ps(pVect2);

        __m256 product = _mm256_mul_ps(v1, v2);
        dot_product_sum = _mm256_add_ps(dot_product_sum, product);

        __m256 squared_v1 = _mm256_mul_ps(v1, v1);
        squared_sum1 = _mm256_add_ps(squared_sum1, squared_v1);

        __m256 squared_v2 = _mm256_mul_ps(v2, v2);
        squared_sum2 = _mm256_add_ps(squared_sum2, squared_v2);

        pVect1 += 8;
        pVect2 += 8;
    }

    float PORTABLE_ALIGN32 dot_product[8];
    float PORTABLE_ALIGN32 norm1[8];
    float PORTABLE_ALIGN32 norm2[8];

    _mm256_store_ps(dot_product, dot_product_sum);
    _mm256_store_ps(norm1, squared_sum1);
    _mm256_store_ps(norm2, squared_sum2);

    float dot_product_sum_scalar = dot_product[0] + dot_product[1] + dot_product[2] + dot_product[3] +
                                    dot_product[4] + dot_product[5] + dot_product[6] + dot_product[7];
    float norm1_sum_scalar = norm1[0] + norm1[1] + norm1[2] + norm1[3] +
                              norm1[4] + norm1[5] + norm1[6] + norm1[7];
    float norm2_sum_scalar = norm2[0] + norm2[1] + norm2[2] + norm2[3] +
                              norm2[4] + norm2[5] + norm2[6] + norm2[7];

    float cosine_similarity = dot_product_sum_scalar / (sqrt(norm1_sum_scalar) * sqrt(norm2_sum_scalar));
    float cosine_distance = 1.0f - cosine_similarity;

    return cosine_distance;
}

static float CosineDistanceAVX2(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {  
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 dot_product_sum1 = _mm256_setzero_ps();
    __m256 dot_product_sum2 = _mm256_setzero_ps();
    __m256 squared_sum1 = _mm256_setzero_ps();
    __m256 squared_sum2 = _mm256_setzero_ps();

    while (pVect1 < pEnd1) {
        __m256 v1_1 = _mm256_loadu_ps(pVect1);
        __m256 v2_1 = _mm256_loadu_ps(pVect2);

        __m256 v1_2 = _mm256_loadu_ps(pVect1 + 8);
        __m256 v2_2 = _mm256_loadu_ps(pVect2 + 8);

        __m256 product1 = _mm256_mul_ps(v1_1, v2_1);
        __m256 product2 = _mm256_mul_ps(v1_2, v2_2);

        dot_product_sum1 = _mm256_add_ps(dot_product_sum1, product1);
        dot_product_sum2 = _mm256_add_ps(dot_product_sum2, product2);

        __m256 squared_v1_1 = _mm256_mul_ps(v1_1, v1_1);
        __m256 squared_v2_1 = _mm256_mul_ps(v2_1, v2_1);
        __m256 squared_v1_2 = _mm256_mul_ps(v1_2, v1_2);
        __m256 squared_v2_2 = _mm256_mul_ps(v2_2, v2_2);

        squared_sum1 = _mm256_add_ps(squared_sum1, _mm256_add_ps(squared_v1_1, squared_v2_1));
        squared_sum2 = _mm256_add_ps(squared_sum2, _mm256_add_ps(squared_v1_2, squared_v2_2));

        pVect1 += 16;
        pVect2 += 16;
    }

    __m256 dot_product_sum = _mm256_add_ps(dot_product_sum1, dot_product_sum2);
    __m256 squared_sum = _mm256_add_ps(squared_sum1, squared_sum2);

    alignas(32) float dot_product[8];
    alignas(32) float squared_sum_arr[8];
    _mm256_store_ps(dot_product, dot_product_sum);
    _mm256_store_ps(squared_sum_arr, squared_sum);

    float dot_product_sum_scalar = 0.0f;
    float squared_sum_scalar = 0.0f;
    for (int i = 0; i < 8; ++i) {
        dot_product_sum_scalar += dot_product[i];
        squared_sum_scalar += squared_sum_arr[i];
    }
    
    float cosine_similarity = dot_product_sum_scalar / (sqrt(squared_sum_scalar) * sqrt(squared_sum_scalar));
    float cosine_distance = 1.0f - cosine_similarity;

    return cosine_distance;
}
static float
CosineDistanceSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2;

    const float *pEnd1 = pVect1 + (qty4 << 2);

    __m128 dot_product_sum = _mm_setzero_ps();
    __m128 squared_sum1 = _mm_setzero_ps();
    __m128 squared_sum2 = _mm_setzero_ps();

    while (pVect1 < pEnd1) {
        __m128 v1 = _mm_loadu_ps(pVect1);
        __m128 v2 = _mm_loadu_ps(pVect2);

        __m128 product = _mm_mul_ps(v1, v2);
        dot_product_sum = _mm_add_ps(dot_product_sum, product);

        __m128 squared_v1 = _mm_mul_ps(v1, v1);
        squared_sum1 = _mm_add_ps(squared_sum1, squared_v1);

        __m128 squared_v2 = _mm_mul_ps(v2, v2);
        squared_sum2 = _mm_add_ps(squared_sum2, squared_v2);

        pVect1 += 4;
        pVect2 += 4;
    }

    float PORTABLE_ALIGN32 dot_product[4];
    float PORTABLE_ALIGN32 norm1[4];
    float PORTABLE_ALIGN32 norm2[4];

    _mm_store_ps(dot_product, dot_product_sum);
    _mm_store_ps(norm1, squared_sum1);
    _mm_store_ps(norm2, squared_sum2);

    float dot_product_sum_scalar = dot_product[0] + dot_product[1] + dot_product[2] + dot_product[3];
    float norm1_sum_scalar = norm1[0] + norm1[1] + norm1[2] + norm1[3];
    float norm2_sum_scalar = norm2[0] + norm2[1] + norm2[2] + norm2[3];

    float cosine_similarity = dot_product_sum_scalar / (sqrtf(norm1_sum_scalar) * sqrtf(norm2_sum_scalar));
    float cosine_distance = 1.0f - cosine_similarity;

    return cosine_distance;
}

#endif