#include <stdint.h>
#include <stdio.h>
#include <zlib.h>
#include "ipfix.h"

typedef AppFlowValues_t Values;
typedef AppsCollection_t Collection;

uint32_t fwidth_hours = sizeof(Collection);

uint32_t foffset_hours = (uint32_t)((uint64_t)(&(((Collection*)0)->outbytes)));

#define FLOW_INIT_COLLECTTION(CUR, VALS) \
    (CUR)->next = 0;					 \
    (CUR)->values = *(VALS); 	   		 \
    (CUR)->inbytes = 0;			   		 \
    (CUR)->inpackets = 0;				 \
	(CUR)->outbytes = 0;			     \
	(CUR)->outpackets = 0;

#define MKFLOW_CRC(VALS)  (VALS)->crc

#include "flowgen.h"

int fexporter_hours(uint32_t exporter){
    return 1;
}

void fcheck_hours(const ipfix_query_buf_t* buf, const ipfix_query_info_t* info, ipfix_query_pos_t* poses){
    Collection* collect;
    Values vals;
    const ipfix_apps_t* apps = info->apps;
    const ipfix_apps_t* oneapp;
    uint32_t minrefs = info->minrefs;
    FlowAppCallback callback = info->callback;
    ReduxCallback redux = info->redux;
    void* callobj = info->callobj;
    const ipfix_app_counts_t* firstcount = (ipfix_app_counts_t*)info->entries;
    const ipfix_app_flow_t* firstflow = info->appflows;

    while(poses->countpos < info->count){
        const ipfix_app_counts_t* counters = firstcount+poses->countpos;
        const ipfix_app_flow_t* flowentry = firstflow + counters->appindex;
        {
        	oneapp = apps+flowentry->app.application;
        	if(oneapp->refcount <= minrefs){
        		redux(callobj, (ipfix_app_counts_t*)counters, flowentry, oneapp);
        	}

        	vals.pos = counters->appindex;

        	callback(callobj, flowentry, &vals);

            collect = lookup(buf, &vals, poses);
            if(collect == NULL){
                return;
            }
			collect->inbytes += counters->inbytes;
			collect->inpackets += counters->inpackets;
			collect->outbytes += counters->outbytes;
			collect->outpackets += counters->outpackets;
        }

        poses->countpos++;
    }
}
