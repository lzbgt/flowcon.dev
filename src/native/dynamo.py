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

def main():
    fields = {'130': ['1.2.3.4/24', '1.2.4.6', '*']}

    top = os.path.join(os.path.dirname(__file__), '..', '..', 'cython')
    loc = os.path.join(top, 'gen')
    incs = os.path.join(top, '..', 'includes')
    fname = os.path.join(loc, 'test.c')
    

    qid, css, lss = validate(fields)
    
    mname = 'Q_'+qid
    modfile = os.path.join(loc, mname+'.so')
    if not os.path.isfile(modfile):
        gensource(fname, qid, css, lss)
        build(incs, loc, fname, qid, mname)
    
    rq = qmod.RawQuery(os.path.join(loc, modfile), qid)
    
    rq.testflow(0x01020406)

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
                lss.add(tofield(fk))
            else:
                css[fk] = [_mkone(ftype, fv)]
        else:
            ss = set()
            for v in fv:
                v = v.strip()                
                if v == '*':
                    lss.add(tofield(fk))
                else:
                    ss.add(_mkone(ftype, v))
            css[fk] = sorted(ss)

    # all fields we need to consider for aggregation
    # (except counters fields) and report, i.e. all `*` fields
    ls = ['%d'%x for x in sorted(lss)]
    res = []
    for fk in sorted(css.keys()):
        res.append((fk, css[fk]))
    res.extend(ls)

    sha = hashlib.sha1()
    sha.update(json.dumps(res))
    qid = sha.hexdigest()
    return qid, css, ls
    
def gensource(fname, qid, css, lss):
    with open(fname, 'w') as f:
        writehead(f, qid)
        # fields to filter flows with
        # If these fields no not match with given flow then flow is discarded
        for fk in sorted(css.keys()):
            lns = css[fk]
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
        writetail(f)
    return qid

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

def writehead(f, qid):
    f.write('#include <stdint.h>\n')
    f.write('#include "ipfix.h"\n')
    f.write("\n")
    f.write("int fcheck_%s(const ipfix_flow_t* flow){\n"%(qid))

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
    f.write("    return 1;\n")
    f.write("}\n")

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