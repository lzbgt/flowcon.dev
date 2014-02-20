#include <stdint.h>
#include <stdio.h>
#include <zlib.h>
#include "ipfix.h"

typedef AppFlowValues_t Values;
typedef AppsCollection_t Collection;

uint32_t fwidth_minutes = sizeof(Collection);

uint32_t foffset_minutes = (uint32_t)((uint64_t)(&(((Collection*)0)->outbytes)));

#define FLOW_INIT_COLLECTTION(CUR, VALS) \
    (CUR)->next = 0;					 \
    (CUR)->values = *(VALS); 	   		 \
    (CUR)->inbytes = 0;			   		 \
    (CUR)->inpackets = 0;				 \
	(CUR)->outbytes = 0;			     \
	(CUR)->outpackets = 0;

#define MKFLOW_CRC(VALS)  (VALS)->crc

#include "flowgen.h"

int fexporter_minutes(uint32_t exporter){
    return 1;
}

void fcheck_minutes(const ipfix_query_buf_t* buf, const ipfix_query_info_t* info, ipfix_query_pos_t* poses){
    Collection* collect;
    int ingress;
    Values vals;
    FlowAppCallback callback = info->callback;
    void* callobj = info->callobj;
    const ipfix_store_counts_t* firstcount = info->first;
    const ipfix_store_flow_t* firstflow = info->flows;

    while(poses->countpos < info->count){
        const ipfix_store_counts_t* counters = firstcount+poses->countpos;
        const ipfix_store_flow_t* flowentry = firstflow + counters->flowindex;

        {
        	ingress = callback(callobj, flowentry, &vals);

            collect = lookup(buf, &vals, poses);
            if(collect == NULL){
                return;
            }
            if(ingress){
				collect->inbytes += counters->bytes;
				collect->inpackets += counters->packets;
            } else {
				collect->outbytes += counters->bytes;
				collect->outpackets += counters->packets;
            }
        }

        poses->countpos++;
    }
}

size_t freport_minutes(const ipfix_query_pos_t* totals,
						int accending,
						const void* buf,
						uint32_t count,
						char* out,
						size_t maxsize,
						rep_callback_t callback,
						void* obj){
	return 0;
}

