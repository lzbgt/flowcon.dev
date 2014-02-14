#ifndef TIME_GEN_H_
#define TIME_GEN_H_

#include <string.h>

static inline Collection* lookup(const ipfix_query_buf_t* buf, const Values* vals, ipfix_query_pos_t* poses){
	Collection* collect = (Collection*)buf->data;
    Collection* current;

    /* check previous stamp */
    if(poses->curpos > 0){
		current = collect + poses->curpos;
		if(current->values.stamp == vals->stamp){
			return current;
		}
		if(current->values.stamp > vals->stamp){
			poses->curpos = 0;
		}
		poses->curpos++;
		while(poses->curpos < poses->bufpos){
			current = collect + poses->curpos;
			if(current->values.stamp >= vals->stamp){
				return current;
			}
			poses->curpos++;
		}
    }
	if(poses->bufpos >= buf->count){
		return NULL;	/* no more space left */
	}
	poses->curpos = poses->bufpos;
	current = collect + poses->bufpos;
	current->values = *vals;
    current->bytes = 0;
    current->packets = 0;
	poses->bufpos++;
	return current;
}

#include "comgen.h"
#include <time.h>

#endif /* TIME_GEN_H_ */
