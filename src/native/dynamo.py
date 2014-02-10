'''
Created on Jan 27, 2014

@author: schernikov
'''

import os, re, hashlib, json
from distutils.core import setup, Extension
from Cython.Build import cythonize

import native

qmod = native.loadmod('nquery')

import query

ipmaskre = re.compile('(?P<b0>\d{1,3})\.(?P<b1>\d{1,3})\.(?P<b2>\d{1,3})\.(?P<b3>\d{1,3})/(?P<mask>\d{1,2})$')

class Builder(object):

    def __init__(self):
        self.top = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'cython')))
        self.loc = os.path.join(self.top, 'gen')
        self.incs = os.path.join(self.top, '..', 'includes')

    def build(self, vres, gensource, cls, pref):
        qid, css, lss, s = vres
        mname = '%s_%s'%(pref, qid)
        fname = os.path.join(self.loc, '%s.c'%(mname))
        modfile = os.path.join(self.loc, mname+'.so')
        if not os.path.isfile(modfile):
            with open(fname, 'w') as f:
                gensource(f, qid, css, lss, s)
            build(self.incs, self.loc, fname, qid, mname)
        
        return cls(os.path.join(self.loc, modfile), qid)
    
dynbuilder = Builder()
    
def genraw(fields):
    vres = validate(fields)
    return dynbuilder.build(vres, genrawsource, qmod.RawQuery, 'R')

def genper(fields):
    expnm = '130'
    exporter = fields.get(expnm, None)
    lss = set()
    if exporter is not None:
        #for k, v in fields
        flds = fields.copy()
        del flds[expnm]
        cs = valfield(expnm, exporter, '', lss)
        vres = validate(flds, expcs = cs, expls = lss, expnm = expnm)
    else:
        cs = []
        vres = validate(fields)

    def gensource(*args):
        genpersource(cs, lss, *args)
    
    return dynbuilder.build(vres, gensource, qmod.PeriodicQuery, 'P')

   
def genpersource(explns, explss, f, qid, css, lss, s):
    writehead(f, s)
    
    colltypename = 'Collection'
    
    f.write("typedef struct PACKED {\n")
    
    flowtypes = []
    attrtypes = []
    exptypes = []
    for ftype in lss:
        if ftype in query.Query.specialtuple:
            continue
        f.write("    uint%d_t %s;\n"%(ftype.size*8, ftype.name))
        if ftype in query.Query.flowtuple:
            flowtypes.append(ftype)
        else:
            attrtypes.append(ftype)

    for expftype in explss:
        f.write("    uint%d_t %s;\n"%(expftype.size*8, expftype.name))
        exptypes.append(expftype)

    f.write("} Values;\n")

    writestructs(f, qid, colltypename)
    
    f.write('#include "flowgen.h"\n\n')
    
    flowchecks = {}
    attrchecks = {}
    for ck, cv in css.items():
        ftype = query.ntypes.Type.all.get(ck, None)
        if not ftype: continue
        if ftype in query.Query.specialtuple: continue
        if ftype in query.Query.flowtuple:
            flowchecks[ck] = cv
        else:
            attrchecks[ck] = cv

    if flowchecks:
        writelocalhead(f, 'int check_flow_tuple', 'const ipfix_flow_tuple_t* entry');
        wcondgroup(f, flowchecks)

    if attrchecks:
        writelocalhead(f, 'int check_flow_attr', 'const ipfix_attributes_t* entry');
        wcondgroup(f, attrchecks)
    
    wfunchead(f, qid, 'int fexporter', "uint32_t exporter")
    wcondition(f, explns)
    writecheckerend(f);

    wfunchead(f, qid, 'void fcheck', "const ipfix_query_buf_t* buf",
                                     "const ipfix_query_info_t* info",
                                     "ipfix_query_pos_t* poses",
                                     "uint32_t exporter")
    f.write("    %s* collect;\n    Values vals;\n"%(colltypename))
    f.write("    const ipfix_store_counts_t* firstcount = info->first;\n")
    if flowtypes or attrtypes:
        f.write("    const ipfix_store_flow_t* firstflow = info->flows;\n")
        if attrtypes:
            f.write("    const ipfix_store_attributes_t* firstattr = info->attrs;\n")
    f.write("\n")
    if not flowtypes and not attrtypes and not exptypes:
        wlookupcall(f, "    ")
    
    f.write("""    while(poses->countpos < info->count){\n""")
    f.write("        const ipfix_store_counts_t* counters = firstcount+poses->countpos;\n")
    if flowtypes or attrtypes or flowchecks or attrchecks:
        f.write("        const ipfix_store_flow_t* flowentry = firstflow + counters->flowindex;\n")
        if attrtypes or attrchecks:
            f.write("        const ipfix_attributes_t* attr = &((firstattr + flowentry->attrindex)->attributes);\n")
        if flowtypes or flowchecks:
            f.write("        const ipfix_flow_tuple_t* flow = &flowentry->flow;\n")
        f.write("\n")
        if flowchecks and attrchecks:
            f.write("        if(check_flow_tuple(flow) && check_flow_attr(attr))")
        elif flowchecks:
            f.write("        if(check_flow_tuple(flow))")
        elif attrchecks:
            f.write("        if(check_flow_attr(attr))")
        else:
            f.write("        ")
    else:
        f.write("        ")
    f.write("{\n")
    if flowtypes or attrtypes or exptypes:
        for ftp in flowtypes:
            f.write("            vals.%s = flow->%s;\n"%(ftp.name, ftp.name))
        for atp in attrtypes:
            f.write("            vals.%s = attr->%s;\n"%(atp.name, atp.name))
        for etp in exptypes:
            f.write("            vals.%s = %s;\n"%(etp.name, etp.name))
        wlookupcall(f, "            ")
    f.write("""
            collect->bytes += counters->bytes;
            collect->packets = counters->packets;
        }

        poses->countpos++;
    }\n""")
    writefunctail(f)
    
    wfunchead(f, qid, 'size_t freport', "const void* buf", "uint32_t count", 
                      "char* out","size_t maxsize", "rep_callback_t callback", "void* obj")
    f.write("""
    uint32_t i;
    uint64_t totbytes=0, totpackets=0;
    int num;
    Collection* collection = (Collection*)buf;
    size_t size = maxsize;\n""")
    tlst = []
    hasip = False
    if lss or explss:
        tlst.extend(lss)
        tlst.extend(explss)
        tlst = sorted(tlst, key=lambda l: l.id)
        for tp in tlst:
            if type(tp) == query.ntypes.IPType: hasip = True
    if hasip:            
        f.write("    unsigned char* pip;\n")
    f.write('\n')
    wsnprintf(f, "    ", r'"{\"counts\":["')
    f.write("    for (i = 0; i < count; ++i) {\n")
    f.write("        if(i == 0) {\n")
    wsnprintf(f, "        ", '"[["')
    f.write("        } else {\n")
    wsnprintf(f, "        ", '",[["')
    f.write("        }\n")
    totids = ''
    sep = ''
    for tp in tlst:
        if type(tp) == query.ntypes.IPType: 
            f.write("        pip = (unsigned char*)&collection->values.%s;\n"%(tp.name))
            wsnprintf(f, "        ", '"'+sep+r'\"%d.%d.%d.%d\"", pip[3], pip[2], pip[1], pip[0]')
        else:
            wsnprintf(f, "        ", '"'+sep+r'"\"%d\"", collection->values.'+tp.name)
        totids += r'%s\"%s\"'%(sep, tp.id)
        sep = ','
    wsnprintf(f, "        ", r'"],[%llu,%llu]]", (LLUT)collection->bytes, (LLUT)collection->packets')
    f.write("""
        totbytes += collection->bytes;
        totpackets += collection->packets;
        collection++;    
    }\n""")
    
    wsnprintf(f, "    ", r'"],\"totals\":{\"counts\":[[%s],'%(totids)+
                         r'[%llu,%llu]", (LLUT)totbytes, (LLUT)totpackets')
    wsnprintf(f, "    ", r'"],\"entries\":%d}}", count')
    f.write('    return maxsize-size;\n')    
    writefunctail(f)

def wsnprintf(f, off, pargs):
    f.write("%sSNPRINTF(%s);\n"%(off, pargs))

def wlookupcall(f, off):
    f.write("""
%scollect = lookup(buf, &vals, poses);
%sif(collect == NULL){
%s    return;
%s}"""%(off, off, off, off))

def writestructs(f, qid, nm):
    f.write("""
typedef struct PACKED %s_t %s;

struct PACKED %s_t{
    uint32_t     next;
    Values       values;
    uint64_t     bytes;
    uint64_t     packets;
};
    """%(nm, nm, nm))
    f.write("\n")
    f.write('uint32_t fwidth_%s = sizeof(%s);\n'%(qid, nm))
    f.write("\n")


def writelocalhead(f, nm, arg):
    f.write("static inline %s(%s){\n"%(nm, arg))

def writefunctail(f):
    f.write("}\n")
    f.write("\n")
    
def writecheckerend(f):
    f.write("    return 1;\n")
    writefunctail(f)

def valfield(fk, fv, pref, lss):
    ftype = query.ntypes.Type.all.get(fk, None)
    if ftype is None:
        raise Exception("Don't know what to do with field %s."%(fk))

    if isone(fv):
        fv = fv.strip()            
        if fv == '*': 
            lss.add(ftype)
            return []
        return [_mkone(pref, ftype, fv)]
    else:
        ss = set()
        for v in fv:
            v = v.strip()                
            if v == '*':
                lss.add(ftype)
            else:
                ss.add(_mkone(pref, ftype, v))
        return sorted(ss)
            
def validate(fields, expcs = None, expls = None, expnm = ''):
    lss = set()
    css = {}
    pref = 'entry->'
    for fk, fv in fields.items():
        cs = valfield(fk, fv, pref, lss)
        if cs: css[fk] = cs

    # all fields we need to consider for aggregation
    # (except counters fields) and report, i.e. all `*` fields
    lss = sorted(lss, key=lambda l: l.id)
    ls = ['%s'%(x.id) for x in lss]
    res = []
    if expcs:
        res.append((expnm, expcs))
    for fk in sorted(css.keys()):
        res.append((fk, css[fk]))
    if expls:
        res.extend(['%s'%(x.id) for x in expls])
    res.extend(ls)

    sha = hashlib.sha1()
    s = json.dumps(res)
    sha.update(s)
    qid = sha.hexdigest()
    return qid, css, lss, s
    
def wfunchead(f, qid, fn, *args):
    s = "%s_%s("%(fn, qid)
    f.write(s)
    off = ' '*len(s)
    if len(args) > 0:
        for a in args[:-1]:
            f.write("%s, \n%s"%(a, off))
        f.write("%s){\n"%(args[-1]))
    
def wcondition(f, lns):
    if len(lns) == 1:
        ln = lns[0]
        if ln:
            f.write("    if (%s) { return 0; }\n"%(ln))
    elif len(lns) > 1:
        s = "    if ("
        f.write(s)
        for ln in lns[:-1]:
            if not ln: continue
            f.write('(%s)'%(ln))
            f.write(' &&\n')
            f.write(' '*len(s))
        f.write("(%s)) { return 0; }\n"%(lns[-1]))
    
def wcondgroup(f, cgroup):
    for fk in sorted(cgroup.keys()):
        lns = cgroup[fk]
        wcondition(f, lns)
    writecheckerend(f)
    
def genrawsource(f, qid, css, lss, s):
    writehead(f, s)
    # fields to filter flows with
    # If these fields no not match with given flow then flow is discarded
    f.write("int fcheck_%s(const ipfix_flow_t* entry){\n"%(qid))
    wcondgroup(f, css)

    iptypes = []
    for ftype in lss:
        if type(ftype) == query.ntypes.IPType:
            if not iptypes: writefromip(f)
            iptypes.append(ftype)

    f.write("void freport_%s(const ipfix_flow_t* flow, char* buf, size_t size){\n"%(qid))

    if iptypes:
        for ftype in lss:
            f.write("    char ip_%s[100];\n"%(ftype.name))
    form = ''
    vals = []
    sep = ''
    for ftype in lss:
        if type(ftype) == query.ntypes.IPType:
            vals.append('fromip(flow->%s, ip_%s, sizeof(ip_%s))'%(ftype.name, ftype.name, ftype.name))
            form += sep+'\\"%s\\"'
        else:
            vals.append('flow->%s'%(ftype.name))
            form += sep+'\\"%d\\"'
        sep = ', '
    pref = "    snprintf(buf, size, "
    off = ' '*len(pref)
    f.write('%s"[[%s],[%%d,%%d]]", \n'%(pref, form))
    for v in vals:
        f.write('%s%s,\n'%(off, v))
    f.write('%sflow->bytes, flow->packets);\n'%(off))
    writefunctail(f)

    writetail(f)
    return qid

def writefromip(f):
    f.write("static const char* fromip(uint32_t ip, char* buf, size_t size){\n")
    f.write("    unsigned char* pip = (unsigned char*)&ip;\n")
    f.write('    snprintf(buf, size, "%d.%d.%d.%d", pip[3], pip[2], pip[1], pip[0]);\n')
    f.write("    return buf;\n")
    writefunctail(f)

def toint(fk):
    try:
        return int(fk)
    except:
        raise Exception("Invalid value '%s' expected integer"%(fk))
    
def tofield(fk):
    try:
        return int(fk)
    except:
        raise Exception("Invalid field number '%s'"%(fk))
    
def ipvariations(value):
    m = ipmaskre.match(value)
    if not m: return None
    dd = m.groupdict()
    mask = int(dd['mask'])
    if mask <= 0:
        return []
    if mask >= 32:
        return None
    ipval = 0
    for bn in range(4):
        b = int(dd['b%d'%bn])
        ipval <<= 8
        ipval += b
    msk = 2**(32-mask)-1
    nmsk = (2**32-1) & (~msk)
    mn = ipval & nmsk
    mx = mn | msk
    return mn, mx

def writehead(f, s):
    f.write('#include <stdint.h>\n')
    f.write('#include <stdio.h>\n')
    f.write('#include <zlib.h>\n')
    f.write('#include "ipfix.h"\n')
    f.write("\n")
    f.write('/*\n %s \n*/'%(s))
    f.write("\n")

def _mkone(pref, ftype, v):
    if type(ftype) == query.ntypes.IPType:
        res = ipvariations(v)
        if res is None: # check exact value match
            try:
                return "%s%s != 0x%x"%(pref, ftype.name, ftype.convert(v))
            except:
                raise Exception("Expected IP got '%s'"%(v))
        if not res:     # any will match
            return ""
        mn, mx = res
        return "%s%s < 0x%x || %s%s > 0x%x"%(pref, ftype.name, mn, pref, ftype.name, mx)
    
    return "%s%s != 0x%x"%(pref, ftype.name, toint(v)) 

def writetail(f):
    f.write('\n')

def isone(fv):
    try:
        iter(fv)
        if isinstance(fv, basestring):
            return True
        else:
            return False
    except TypeError:
        pass
    return True


def build(includes, sources, fname, qid, target):
    cd = os.getcwd()
    try:
        os.chdir(os.path.dirname(fname))
        buildc(fname, includes, qid, target)
    finally:
        os.chdir(cd)

def buildc(fname, includes, qid, nm):
    setup(ext_modules = [Extension(nm,
                                   sources=[os.path.basename(fname)], 
                                   include_dirs=[includes])],
          script_args=['build_ext', '--inplace'])

def buildcython(fname, includes):
    setup(ext_modules = cythonize([os.path.basename(fname)], include_path=[includes]),
          script_args=['build_ext', '--inplace'])
