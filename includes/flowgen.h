#ifndef FLOW_GEN_H_
#define FLOW_GEN_H_

#include <string.h>

#ifndef FLOW_INIT_COLLECTTION

#define FLOW_INIT_COLLECTTION(CUR, VALS) \
    (CUR)->next = 0;					 \
    (CUR)->values = *(VALS); 	   		 \
    (CUR)->bytes = 0;			   		 \
    (CUR)->packets = 0;

#endif //FLOW_INIT_COLLECTTION

#ifndef MKFLOW_CRC

#define MKFLOW_CRC(VALS)  adler32(1, (void*)(VALS), sizeof(*(VALS)))

#endif //MKFLOW_CRC

static inline Collection* lookup(const ipfix_query_buf_t* buf, const Values* vals, ipfix_query_pos_t* poses){
	uLong crc;
	uint32_t idx, pos, last;
	Collection* collect = (Collection*)buf->data;
    Collection* current;

	crc = MKFLOW_CRC(vals);
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
/*				printf("   ===crc:0x%08x idx:%d pos:%d exp:0x%08x\n", (unsigned int)crc,
						idx, pos, current->values.exporter);*/
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
	FLOW_INIT_COLLECTTION(current, vals);

/*	printf("   +++crc:0x%08x idx:%d pos:%d exp:0x%08x\n", (unsigned int)crc,
			idx, last, current->values.exporter);*/

    return current;
}

#include "comgen.h"

#endif /* FLOW_GEN_H_ */
