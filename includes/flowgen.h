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

    return current;
}

#endif /* FLOW_GEN_H_ */
