#ifndef FLOW_IPFIX_H_
#define FLOW_IPFIX_H_

#define TEMPLATE_SET_ID 2
#define MINDATA_SET_ID 256

#define PACKET_DATA_SIZE 1458 /* ethernet 14, ip 20, udp 8 */
#define IPFIX_HEADER_SIZE 16
#define IPFIX_SET_HEADER_SIZE 4

/* Template buffer cannot be filled more than to packet data size */
#define TEMPLATE_BUFFER_SIZE (PACKET_DATA_SIZE - IPFIX_HEADER_SIZE)

#define PACKED __attribute__((__packed__))

typedef struct PACKED ipfix_header {
	uint16_t version;
	uint16_t length;
	uint32_t exportTime;
	uint32_t sequenceNumber;
	uint32_t observationDomainId;
} ipfix_header_t;

typedef struct PACKED ipfix_template_set_header {
	uint16_t id;
	uint16_t length;
} ipfix_template_set_header_t;

typedef struct PACKED ipfix_flow {
	uint32_t bytes;
	uint32_t packets;
	uint8_t  protocol;
	uint8_t  tos;
	uint8_t  tcpflags;
	uint16_t srcport;
	uint32_t srcaddr;
	uint8_t  srcmask;
	uint32_t inpsnmp;
	uint16_t dstport;
	uint32_t dstaddr;
	uint8_t  dstmask;
	uint32_t outsnmp;
	uint32_t nexthop;
	uint32_t srcas;
	uint32_t dstas;
	uint32_t last;
	uint32_t first;
	uint32_t exporter;
} ipfix_flow_t;

typedef struct PACKED ipfix_flow_tuple {
	uint8_t  protocol;
	uint16_t srcport;
	uint32_t srcaddr;
	uint16_t dstport;
	uint32_t dstaddr;
} ipfix_flow_tuple_t;

typedef struct PACKED ipfix_attributes {
	uint8_t  tos;
	uint8_t  tcpflags;
	uint8_t  srcmask;
	uint32_t inpsnmp;
	uint8_t  dstmask;
	uint32_t outsnmp;
	uint32_t nexthop;
	uint32_t srcas;
	uint32_t dstas;
} ipfix_attributes_t;

typedef uint32_t indextype;

typedef struct PACKED ipfix_store_flow {
	indextype			next;
	uint32_t			crc;
	ipfix_flow_tuple_t 	flow;
	indextype			attrindex;
	uint16_t			refcount;
} ipfix_store_flow_t;

typedef struct PACKED ipfix_store_attributes {
	indextype			next;
	uint32_t			crc;
	ipfix_attributes_t 	attributes;
} ipfix_store_attributes_t;

typedef struct PACKED ipfix_store_entry {
	indextype			next;
	uint32_t			crc;
	char				data[0];
} ipfix_store_entry_t;

typedef struct PACKED ipfix_store_counts {
	indextype		 	flowindex;
	uint32_t			bytes;
	uint32_t			packets;
} ipfix_store_counts_t;

typedef struct PACKED ipfix_query_buf {
	void*				data;
	uint32_t 			count;

	uint32_t*			poses;
	uint32_t			mask;
} ipfix_query_buf_t;

typedef struct PACKED ipfix_query_pos {
	uint32_t 			bufpos;
	uint32_t 			curpos;
	uint32_t 			countpos;

	uint64_t			totbytes;
	uint64_t			totpackets;
} ipfix_query_pos_t;

typedef struct PACKED ipfix_query_info {
	const ipfix_store_counts_t* 	first;
	uint32_t 				  		count;
	const ipfix_store_flow_t* 		flows;
	const ipfix_store_attributes_t* attrs;
	uint64_t						stamp;
	uint32_t 				  		exporter;
} ipfix_query_info_t;

typedef void (*ipfix_collector_call_t)(const ipfix_query_buf_t* buf,
									   const ipfix_query_info_t* info,
									   ipfix_query_pos_t* poses);

typedef char* (*rep_callback_t)(void* data, size_t* size);
typedef size_t (*ipfix_collector_report_t)(const ipfix_query_pos_t* totals, int accending,
										   const void* buf, uint32_t count, char* out,
										   size_t maxsize, rep_callback_t callback, void* obj);

#endif /* FLOW_IPFIX_H_ */
