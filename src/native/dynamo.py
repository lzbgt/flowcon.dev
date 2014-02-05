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
    
def main():
    #fields = {'8': ['1.2.3.4/24', '1.2.4.6', '*'], '12':'*'}
    fields = {'130': ['1.2.3.4/24', '1.2.4.6', '*'], '12':'*'}
    #fields = {'130': ['1.2.3.4/24', '1.2.4.6']}

    rq = genper(fields)
    
    rq.testflow(0x01020406)

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
        s = json.dumps((expnm, exporter))
        cs = valfield(expnm, exporter, '', lss)
        vres = validate(flds)
    else:
        s = ''
        cs = []
        vres = validate(fields)

    def gensource(*args):
        genpersource(cs, lss, s, *args)
    
    return dynbuilder.build(vres, gensource, qmod.PeriodicQuery, 'P')

   
def genpersource(explns, explss, exps, f, qid, css, lss, s):
    "remove field 130 from checker and add static value to filler"
    """
        fcheck_
        freport_
        fexporter_
        fwidth_    
    """
    writehead(f, s+exps)
    
    colltypename = 'Collection'
    
    f.write("typedef struct PACKED {\n")
    
    flownames = []
    attrnames = []
    expnames = []
    
    for ftype in lss:
        if ftype in query.Query.specialtuple:
            continue
        f.write("    uint%d_t %s;\n"%(ftype.size*8, ftype.name))
        if ftype in query.Query.flowtuple:
            flownames.append(ftype.name)
        else:
            attrnames.append(ftype.name)

    for expftype in explss:
        f.write("    uint%d_t %s;\n"%(expftype.size*8, expftype.name))
        expnames.append(expftype.name)

    f.write("} Values;\n")

    writestructs(f, qid, colltypename)

    writelocalhead(f, 'int check_flow_tuple', 'const ipfix_flow_tuple_t* tup');
    writecheckerend(f);
    
    writelocalhead(f, 'int check_flow_attr', 'const ipfix_attributes_t* attr');
    writecheckerend(f);
    
    writelookup(f, colltypename)
        
    wfunchead(f, qid, 'int fexporter', "uint32_t exporter")
    wcondition(f, explns)
    writecheckerend(f);

    wfunchead(f, qid, 'void fcheck', "const ipfix_query_buf_t* buf",
                                     "const ipfix_query_info_t* info",
                                     "ipfix_query_pos_t* poses",
                                     "uint32_t exporter")
    f.write("    %s* collect;\n    Values vals;\n"%(colltypename))
    f.write("    const ipfix_store_counts_t* counters = info->first+poses->countpos;\n")
    if flownames or attrnames:
        f.write("    const ipfix_store_flow_t* firstflow = info->flows;\n")
        if attrnames:
            f.write("    const ipfix_store_attributes_t* firstattr = info->attrs;\n")
    f.write("\n")
    if not flownames and not attrnames and not expnames:
        wlookupcall(f, "    ")
    
    f.write("""    while(poses->countpos < info->count){\n""")
    if flownames or attrnames:
        f.write("        const ipfix_store_flow_t* flowentry = firstflow + counters->flowindex;\n")
        if attrnames:
            f.write("        const ipfix_attributes_t* attr = &((firstattr + flowentry->attrindex)->attributes);\n")
        if flownames:
            f.write("        const ipfix_flow_tuple_t* flow = &flowentry->flow;\n")
        f.write("\n")
        if flownames and attrnames:
            f.write("        if(check_flow_tuple(flow) && check_flow_attr(attr))")
        elif flownames:
            f.write("        if(check_flow_tuple(flow))")
        else:
            f.write("        if(check_flow_attr(attr))")
    else:
        f.write("        ")
    f.write("{\n")
    if flownames or attrnames or expnames:
        for nm in flownames:
            f.write("            vals.%s = flow->%s;\n"%(nm, nm))
        for nm in attrnames:
            f.write("            vals.%s = attr->%s;\n"%(nm, nm))
        for nm in expnames:
            f.write("            vals.%s = %s;\n"%(nm, nm))
        wlookupcall(f, "            ")
    f.write("""
            collect->bytes += counters->bytes;
            collect->packets = counters->packets;
        }

        poses->countpos++;
    }\n""")
    f.write("}\n")
    f.write("\n")

def wlookupcall(f, off):
    f.write("""
%scollect = lookup(buf, &vals, poses);
%sif(collect == NULL){
%s    return;
%s}"""%(off, off, off, off))

def writelookup(f,  nm):
    writelocalhead(f, '%s* lookup'%(nm), 
                   'const ipfix_query_buf_t* buf, const Values* vals, ipfix_query_pos_t* poses');
    f.write('    %s* collect = (%s*)buf->data;\n'%(nm, nm))
    f.write("\n")
    f.write('    poses->bufpos++;\n')
    f.write('    if(poses->bufpos >= buf->count){\n')
    f.write('        return 0;\n')
    f.write("    }\n")
    f.write("    return collect;\n")
    f.write("}\n")
    f.write("\n")

def writestructs(f, qid, nm):
    f.write("""
typedef struct PACKED %s_t %s;

struct PACKED %s_t{
    %s*     next;
    Values          values;
    uint64_t        bytes;
    uint64_t        packets;
};
    """%(nm, nm, nm, nm))
    f.write("\n")
    f.write('uint32_t fwidth_%s = sizeof(%s);\n'%(qid, nm))
    f.write("\n")


def writelocalhead(f, nm, arg):
    f.write("static inline %s(%s){\n"%(nm, arg))

def writecheckerend(f):
    f.write("    return 1;\n")
    f.write("}\n")
    f.write("\n")

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
            
def validate(fields):
    lss = set()
    css = {}
    pref = 'flow->'
    for fk, fv in fields.items():
        cs = valfield(fk, fv, pref, lss)
        if cs: css[fk] = cs

    # all fields we need to consider for aggregation
    # (except counters fields) and report, i.e. all `*` fields
    lss = sorted(lss, key=lambda l: l.id)
    ls = ['%s'%(x.id) for x in lss]
    res = []
    for fk in sorted(css.keys()):
        res.append((fk, css[fk]))
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
    
def genrawsource(f, qid, css, lss, s):
    writehead(f, s)
    # fields to filter flows with
    # If these fields no not match with given flow then flow is discarded
    f.write("int fcheck_%s(const ipfix_flow_t* flow){\n"%(qid))
    for fk in sorted(css.keys()):
        lns = css[fk]
        wcondition(f, lns)
    f.write("    return 1;\n")
    f.write("}\n")
    f.write("\n")
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
    f.write("}\n")
    writetail(f)
    return qid

def writefromip(f):
    f.write("static const char* fromip(uint32_t ip, char* buf, size_t size){\n")
    f.write("    unsigned char* pip = (unsigned char*)&ip;\n")
    f.write('    snprintf(buf, size, "%d.%d.%d.%d", pip[3], pip[2], pip[1], pip[0]);\n')
    f.write("    return buf;\n")
    f.write("}\n")

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

if __name__ == '__main__':
    main()