#ifndef TIME_GEN_H_
#define TIME_GEN_H_

#include <string.h>

static inline Collection* lookup(const ipfix_query_buf_t* buf, const Values* vals, ipfix_query_pos_t* poses){
	uLong crc;
	uint32_t idx, pos, last;
	Collection* collect = (Collection*)buf->data;
    Collection* current;

    /* check previous stamp */
    current = collect + poses->bufpos-1;
    if(current->values.stamp == vals->stamp){
    	if(poses->bufpos > 1){
    		return current;
    	}
    }
	if(poses->bufpos >= buf->count){
		return NULL;	/* no more space left */
	}
	current = collect + poses->bufpos;
	poses->bufpos++;
	return current;
}

#include "comgen.h"

#endif /* TIME_GEN_H_ */
