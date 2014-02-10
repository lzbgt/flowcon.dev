#ifndef FLOW_GEN_H_
#define FLOW_GEN_H_

#include <string.h>

static inline Collection* lookup(const ipfix_query_buf_t* buf, const Values* vals, ipfix_query_pos_t* poses){
	uLong crc;
	uint32_t idx, pos, last;
	Collection* collect = (Collection*)buf->data;
    Collection* current;

	crc = adler32(1, (void*)vals, sizeof(*vals));
	idx = ((uint32_t)crc) & buf->mask;

	pos = buf->poses[idx];
	if(pos == 0){
		last = poses->bufpos;
		if(last >= buf->count){
			return NULL;	/* no more space left */
		}
		buf->poses[idx] = last;
		current = collect + last;
		poses->bufpos++;
	} else {
		do {
			current = collect + pos;

			if(memcmp(&current->values, vals, sizeof(current->values)) == 0){
/*				printf("   ===crc:0x%08x idx:%d pos:%d dst:0x%08x, exp:0x%08x\n", (unsigned int)crc,
						idx, pos, current->values.dstaddr, current->values.exporter);*/
				return current;
			}

			pos = current->next;
		} while(pos != 0);

		last = poses->bufpos;
		if(last >= buf->count){
			return NULL;	/* no more space left */
		}
		current->next = last;
		current = collect + last;
		poses->bufpos++;
	}
    current->next = 0;
    current->values = *vals;
    current->bytes = 0;
    current->packets = 0;

/*	printf("   +++crc:0x%08x idx:%d pos:%d dst:0x%08x, exp:0x%08x\n", (unsigned int)crc,
			idx, last, current->values.dstaddr, current->values.exporter);*/

    return current;
}

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

#endif /* FLOW_GEN_H_ */
