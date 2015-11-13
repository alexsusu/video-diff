import sympy
from sympy.parsing.sympy_parser import parse_expr

from sympy import simplify, cos, sin
from sympy import separatevars
import sympy.simplify
import sympy.simplify.cse_main
from sympy.simplify.cse_main import cse, opt_cse, tree_cse


# See http://docs.sympy.org/dev/_modules/sympy/printing/ccode.html
from sympy.printing import ccode

from sympy.utilities.codegen import codegen
from sympy.abc import x, y, z


defVars = ["xsrc", "ysrc"]
eqList = [None] * len(defVars)



def GenCode(exp):
    # From http://docs.sympy.org/dev/modules/utilities/codegen.html
    [(c_name, c_code), (h_name, c_header)] = codegen(
        ("exp", exp), "C", "test", header=False, empty=False)

    print("c_code = %s" % c_code)
    print("ccode = %s" % ccode(exp))

    #!!!! TODO TODO TODO: we should check if we have symbolic denominator and generate code that takes care to check if the denominator is 0 and if so to abort the compuatation
    return None




"""
From https://groups.google.com/forum/#!msg/sympy/Wqs1OhTBexg/bjYRabHYXOoJ :
  - i.e., sympy mailing list,
      email on Thursday, November 14, 2013 5:56:22 AM UTC+1, Chris Smith wrote.
"""
def eqs2matrix(eqs, unknownSymbols, augment=False):
    eqsMat = sympy.Matrix([ \
                      si.lhs - si.rhs \
                      if isinstance(si, sympy.Equality) else si for si in eqs])
    print("eqs2matrix(): eqsMat = %s" % str(eqsMat))

    # Explanation on what jacobian is: https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Jacobian_determinant
    jcb = eqsMat.jacobian(unknownSymbols)
    print("eqs2matrix(): jcb = %s" % str(jcb))

    rhs = -(eqsMat - jcb * sympy.Matrix(unknownSymbols))
    print("eqs2matrix(): Before simplify, rhs = %s" % str(rhs))
    rhs.simplify()
    print("eqs2matrix(): rhs = %s" % str(rhs))

    if augment:
        # See http://docs.sympy.org/dev/tutorial/matrices.html#deleting-and-inserting-rows-and-columns
        jcbShape = jcb.shape
        res = jcb.col_insert(jcbShape[1], rhs)
    else:
        res = (jcb, rhs)

    return res




def Parse(str):
    print("Parse(): str = %s" % str)

    # Inspired from email mailing list sympy, Jan 29, 2015, Aaron Meurer

    #exp0 = parse_expr("a1*x1**2+a2*x1+a3+a4*x1*x2+a5*x2**2") #, transformations=(auto_number,auto_symbol))
    exp = parse_expr(str)

    print("Parse(): exp = %s" % exp)

    #exp0.subs(a1,10)

    for crtSym in exp.atoms(sympy.Symbol):
        sympy.var(crtSym.name) #__str__())
    if False:
        for crtSym in exp.atoms(): #sympy.Symbol):
            #sympy.var(crtSym. name)
            print crtSym

    """
    auto_symbol doesn't do namespace injection.
      It just means that the parser won't fail with a NameError (undefined names in the namespace that the parser users will be converted to Symbols).
      Your hack works (I would use each_sym.name instead of __str__(), and note that str(each_sym) is the same as each_sym.__str__()).
    """

    #print exp.subs(cosTheta, 0.1)
    return exp


"""
We get rid, if possible of fractions in the input equations - this keeps
      equations symbolically linear, which makes simplifycation of expressions
      give better final results.
"""
def PreprocessEq(lhs, rhs):
    if rhs.is_Mul:
        args = rhs.as_ordered_factors()
        print("args = %s" % args)
        print("args[1] = %s" % args[1])
        if args[1].is_Pow:
            base = args[1].args[0]
            exponent = args[1].args[1]
            if exponent == -1:
                # rhs is a fraction, more exactly it is of form exp * (1/denominator)
                denominator = base
                lhs = lhs * denominator
                rhs = rhs * denominator

    print("PreprocessEq(): lhs = %s" % lhs)
    print("PreprocessEq(): rhs = %s" % rhs)

    return lhs, rhs

rstripStr = " \n"
def Read(fileName):
    fin = open(fileName)

    while (True):
        str = fin.readline()
        if str:
            #pass
            str = str.rstrip(rstripStr)
        else:
            break

        if str.startswith("#"):
            continue

        for index, v in enumerate(defVars):
            cmpStr = v + " = "
            #print("cmpStr = %s" % cmpStr)
            #" - v"
            if str.startswith(cmpStr):
                #TestTraverseExpr(tmpExp)

                tmpExp = Parse(str[len(cmpStr) : ])
                lhs = sympy.Symbol(v)
                print (tmpExp - lhs)
                lhs, rhs = PreprocessEq(lhs, tmpExp)

                #assert False

                #eqList[index] = Parse(str[len(cmpStr) : ] + " - " + v)
                eqList[index] = rhs - lhs

                #print("eqList[index] = %s" % str(eqList[index]))
                print("eqList[%d] = %s" % (index, eqList[index]))
                #print(separatevars(expr=eqList[index], symbols=[xdst, xsrc]))
                print("Number of operators: %s" % \
                        sympy.count_ops(eqList[index], visual=True))
                GenCode(eqList[index])

        #res = Parse(str)


def TestEqs2Matrix():
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    c = sympy.Symbol("c")
    #eqs = [x + 2*y == 4, 2*c + y/2 == 0]
    eqsCrt = [x + 2*y - 4, 2*c + y/2]
    """
    If augment==False, the first matrix returned is the one of the coef of the
      unknowns, the second of the constants.
      Since I want the 2 matrices concatenated I give augment=True.
    """
    res = eqs2matrix(eqs=eqsCrt, unknownSymbols=(x, c), augment=True)
    print("res = %s" % str(res))
    """
    Example:
        >>> s
        [x + 2*y == 4, 2*c + y/2 == 0]
        >>> eqs2matrix(s, (x, c))
        (Matrix([
        [1, 0],
        [0, 2]]), Matrix([
        [-2*y + 4],
        [    -y/2]]))
        >>> eqs2matrix([2*c*(x+y)-4],(x, y))
        (Matrix([[2*c, 2*c]]), Matrix([[4]]))
    """

def TestTraverseExpr(tmpExp):
    # Inspired from sympy-master/sympy/core/expr.py:

    assert isinstance(tmpExp, sympy.Expr) #fraction)  #Equality)
    assert tmpExp.is_Mul #fraction)  #Equality)
    print("Number of operators on the rhs exp: %s" % \
                sympy.count_ops(tmpExp, visual=True))
    args = tmpExp.as_ordered_factors() #order=order)
    print("args = %s" % args)
    print("args[1] = %s" % args[1])
    assert args[1].is_Pow
    args1 = args[1].args
    print args1
    print args1[0]
    assert args1[1] == -1
    #print("args1 (the denominator factors) = %s" % str(args1))
    #print type(args[1])
    assert args[1].is_Mul


def Main():
    #Read("spec.in")
    #Read("spec_warp_simpler.in")
    #Read("spec_warpPerspective.in")
    Read("spec_rotate.in")

    if False:
        [(c_name, c_code), (h_name, c_header)] = codegen(
            [("xdst", eqList[0]), ("ydst", eqList[1])], "C", "test", header=False, empty=False)

        print("c_code = %s" % c_code)

    print("eqList = %s" % str(eqList))
    resEqSysMat = eqs2matrix(eqs=eqList, unknownSymbols=(xdst, ydst), augment=True)
    print("resEqSysMat = %s" % str(resEqSysMat))

    from sympy import Matrix, solve_linear_system
    res = solve_linear_system(resEqSysMat, xdst, ydst)
    print("The result is (xdst, ydst) = %s" % str(res))
    print("res[xdst] = %s" % str(res[xdst]))
    print("res[ydst] = %s" % str(res[ydst]))

    #sympy.simplify.cse_main.cse(res[xdst], res[ydst])
    expListWithBoundedVars = cse([res[xdst], res[ydst]])
    print("After performing CSE, we have: %s" % str(expListWithBoundedVars))

    print("expListWithBoundedVars[0] (the bounded vars) = %s" % str(expListWithBoundedVars[0]))
    print("expListWithBoundedVars[1][0] = %s" % str(expListWithBoundedVars[1][0]))
    eFinal = []
    for e in expListWithBoundedVars[0]:
        eFinal.append( (str(e[0]), e[1]) )

    #expListWithBoundedVars[0] + \
    expListWithBoundedVars = eFinal + \
                            [("xdst", expListWithBoundedVars[1][0])] + \
                            [("ydst", expListWithBoundedVars[1][1])]
    print("expListWithBoundedVars = %s" % str(expListWithBoundedVars))
    [(c_name, c_code), (h_name, c_header)] = codegen(
        expListWithBoundedVars, "C", "final_test", header=False, empty=False)

    print("c_code = %s" % c_code)
    #!!!!TODO TODO TODO TODO TODO: use ccode to better gen code - just 1 function computing everthing


#TestEqs2Matrix()
Main()



# From http://docs.sympy.org/dev/modules/parsing.html#parsing-functions-reference
#print parse_expr("1/2")

