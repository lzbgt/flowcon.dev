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
    fields = {'130': ['1.2.3.4/24', '1.2.4.6', '*'], '12':'*'}
    #fields = {'130': ['1.2.3.4/24', '1.2.4.6']}

    rq = genper(fields)
    
    rq.testflow(0x01020406)

def genraw(fields):
    vres = validate(fields)
    return dynbuilder.build(vres, genrawsource, qmod.RawQuery, fields, 'R')

def genper(fields):
    vres = validate(fields)
    return dynbuilder.build(vres, genpersource, qmod.PeriodicQuery, fields, 'P')

def validate(fields):
    lss = set()
    css = {}
    for fk, fv in fields.items():
        ftype = query.ntypes.Type.all.get(fk, None)
        if ftype is None:
            raise Exception("Don't know what to do with field %s."%(fk))

        if isone(fv):
            fv = fv.strip()            
            if fv == '*': 
                lss.add(ftype)
            else:
                css[fk] = [_mkone(ftype, fv)]
        else:
            ss = set()
            for v in fv:
                v = v.strip()                
                if v == '*':
                    lss.add(ftype)
                else:
                    ss.add(_mkone(ftype, v))
            css[fk] = sorted(ss)

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
    
def genpersource(f, qid, css, lss, s):
    "remove field 130 from checker and add static value to filler"
    """
        fcheck_
        freport_
        fexporter_
        fwidth_    
    """
    writehead(f, s)
    wfunchead(f, qid, 'int fexporter', "uint32_t ip")
    
    f.write("    return 1;\n")
    f.write("}\n")
    f.write("\n")

    wfunchead(f, qid, 'void fcheck', "const ipfix_query_buf_t* buf",
                                     "const ipfix_query_info_t* info",
                                     "ipfix_query_pos_t* poses",
                                     "uint32_t expip")
    f.write("}\n")
    f.write("\n")
    
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
    f.write('#include "ipfix.h"\n')
    f.write("\n")
    f.write('/*\n %s \n*/'%(s))
    f.write("\n")

def _mkone(ftype, v):
    if type(ftype) == query.ntypes.IPType:
        res = ipvariations(v)
        if res is None: # check exact value match
            try:
                return "flow->%s != 0x%x"%(ftype.name, ftype.convert(v))
            except:
                raise Exception("Expected IP got '%s'"%(v))
        if not res:     # any will match
            return ""
        mn, mx = res
        return "flow->%s < 0x%x || flow->%s > 0x%x"%(ftype.name, mn, ftype.name, mx)
    
    return "flow->%s != 0x%x"%(ftype.name, toint(v)) 

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