#include <stdint.h>

__device__ void fake_md5Hash(uint32_t* answer, uint8_t* data, uint32_t length, uint32_t *a1, uint32_t *b1, uint32_t *c1, uint32_t *d1) {
  uint8_t target[] = {' ', ' ', ' ', ' ', ' '};
  ssize_t len = sizeof(target);

  if (len != length) {
    *a1 = *b1 = *c1 = *d1 = 0;
    return;
  }

  for (auto i = 0; i < len; ++i) {
    if (data[i] != target[i]) {
      *a1 = *b1 = *c1 = *d1 = 0;
      return;
    }
  }

  *a1 = answer[0];
  *b1 = answer[1];
  *c1 = answer[2];
  *d1 = answer[3];
}
