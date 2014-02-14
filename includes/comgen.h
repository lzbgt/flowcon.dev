#ifndef COM_GEN_H_
#define COM_GEN_H_

#include <string.h>

typedef long long unsigned int LLUT;

#define SNPRINTF(...)     	num = snprintf(out, size, __VA_ARGS__);		\
							if(num <= 0){								\
								return 0;								\
							}											\
							if(num >= size){							\
								size_t printed = maxsize-size;			\
								out[0] = 0;								\
								maxsize = printed+num;					\
								out = callback(obj, &maxsize);			\
								if(out == NULL){						\
									return 0;							\
								}										\
								size = maxsize-printed;					\
								if(size <= num){						\
									return 0;							\
								}										\
								out += printed;							\
								num = snprintf(out, size, __VA_ARGS__);	\
								if(num <= 0 || num >= size){			\
									return 0;							\
								}										\
							}											\
							out += num;									\
							size -= num;

#endif /* COM_GEN_H_ */
