# imports
import random
import re
import itertools
import time

OPS = ["IF", "AND", "OR", "NOT", "IMPLIES"]
QUANTS = ["FORALL", "EXISTS"]


class Node:
    def __init__(self,node_type,node_value):
        self.set_node(node_type,node_value)
        self.children = []

    def append_child(self,child):
        self.children.append(child)

    def set_child_nodes(self,children):
        self.children = children

    def get_children(self):
        return self.children

    def set_node(self,node_type,node_value):
        self.node_type = node_type
        self.node_value = node_value

    def get_node_type(self):return self.node_type
    def get_node_value(self):return self.node_value

    def pop_child(self):
        return self.children.pop()
    def get_text(self):
        return "[" + self.get_node_type() + "] " + self.get_node_value()

    def __str__(self, level=0):
        text = "--" * level + self.get_text() + "\n"
        for child_node in self.children:
            text += child_node.__str__(level + 1)
        return text

def print_tree(root):
    print(root)

# parse the statment to list of arguments and each item in the list
# is a tuple consists of (argument type, argument content)
# then pass it to the function parse tree to return the tree of this arguments
def parse(input):
    statment = input
    # argument types: open and close bracket, operator and symbol
    args = []

    # definsive programming to make the input in the right from
    regex = r'''\(|\)|\[|\]|\-?\d+\.\d+|\-?\d+|[^,(^)\s]+'''
    input = input.replace("\t", " ")
    input = input.replace("\n", " ")
    input = input.replace("\r", " ")
    input = input.lstrip(" ")
    input = input.rstrip(" ")

    prev_arg = next_arg = None
    lines = []
    arg_list = re.findall(regex,input)

    for i in range (len(arg_list)):
        arg = arg_list[i]
        if (i - 1 >= 0):
            prev_arg = arg_list[i - 1]
        if (i + 1 < len(arg_list)):
            next_arg = arg_list[i + 1]

        if (arg == "("):
            arg_name = "open_bracket"
        elif (arg == ")"):
            arg_name = "close_bracket"
        elif prev_arg == "(":
            if (arg in OPS):
                arg_name = "op"
            elif (arg in QUANTS):
                arg_name = "quant"
            else:
                arg_name = "function"
        elif (prev_arg in QUANTS):
            arg_name = "variable"
        elif arg.isalnum():
            arg_name = "symbol"

        arg_tuple = (arg_name, arg)
        args.append(arg_tuple)

    return parse_tree(args)




def parse_tree(args):
    stack = []
    stack_element = None

    for i in range(len(args)):
        current_element = args[i]
        current_element_type = current_element[0]
        current_element_value = current_element[1]

        if (current_element_type == "open_bracket" and current_element_value == "("):
            continue
        elif (current_element_type == "close_bracket" and current_element_value == ")"):
            child_nodes = []
            while True:
                current_node = stack.pop()
                if (current_node.node_type in ["op" , "quant", "function"]):
                    current_child_nodes = current_node.get_children()
                    if (len(current_child_nodes) == 0):
                        break
                child_nodes.append(current_node)
            current_node.set_child_nodes(child_nodes)
            stack.append(current_node)


        else:
            node = Node(current_element_type, current_element_value)
            stack.append(node)


    while (len(stack) > 1):
        childs = []
        childs.append(stack.pop())
        childs.append(stack.pop())
        stack[-1].set_child_nodes(childs)
    return stack.pop()


def implication_elimination(node):
    if (node.node_value == "IMPLIES"):
        condition = node.get_children()[-1]
        node.pop_child()
        negation_node = Node("op","NOT")
        negation_node.append_child(condition)
        node.append_child(negation_node)
        node.node_type = "op"
        node.node_value = "OR"
    children = node.get_children()
    for child in children:
        implication_elimination(child)

    return node


def deMorgan_mvNegation(node):
    if (node.node_value == "NOT"):
        if((node.get_children()[0].node_value == "AND") | (node.get_children()[0].node_value == "OR")):
            children = node.get_children()[0].get_children()
            negation_node1 = Node("op","NOT")
            negation_node2 = Node("op", "NOT")
            negation_node1.append_child(children[1])
            negation_node2.append_child(children[0])
            children = [negation_node1, negation_node2]

            if(node.get_children()[0].node_value == "AND"):
                final_node = Node("op","OR")
            else:
                final_node = Node("op", "AND")

            final_node.set_child_nodes(children)
            node.set_node(final_node.node_type, final_node.node_value)
            node.set_child_nodes(final_node.get_children())

        elif((node.get_children()[0].node_value == "FORALL") | (node.get_children()[0].node_value == "EXISTS")):

            negation_node = Node("op", "NOT")
            negation_node.append_child(node.get_children()[0].get_children()[0])

            if(node.get_children()[0].node_value == "FORALL"):
                final_node = Node("quant", "EXISTS")
            else:
                final_node = Node("quant", "FORALL")

            final_node.append_child(negation_node)
            final_node.append_child(node.get_children()[0].get_children()[1])
            node.set_node(final_node.node_type, final_node.node_value)
            node.set_child_nodes(final_node.get_children())

        elif(node.get_children()[0].node_value == "NOT"):
            final_node = node.get_children()[0].get_children()[0]
            node.set_node(final_node.node_type, final_node.node_value)
            node.set_child_nodes(final_node.get_children())

    children = node.get_children()
    for child in children:
        deMorgan_mvNegation(child)
    return node

def standardize(node, variable_names = {}):

    node_type = node.get_node_type()
    node_value = node.get_node_value()
    children = node.get_children()

    if node_type == "quant":

        child_node = children[-1]

        child_node_type = child_node.get_node_type()
        child_node_value = child_node.get_node_value()

        variable_names[child_node_value] = child_node_value + "_" + str(random.randint(0, 10000))

        child_node.set_node(child_node_type, variable_names[child_node_value])

        children[-1] = child_node

    elif node_type == "function" or node_type == "predicate":
        for i in range(len(_child_nodes)):

            child_node = children[i]
            child_node_type = child_node.get_element_type()
            child_node_value = child_node.get_element_value()

            if child_node_value in variable_names:
                child_node.set_node(child_node_type, variable_names[child_node_value])

            children[i] = _child_node

    node.set_child_nodes(children)

    for i in range(len(children)):
        standardize(children[i], variable_names)

    return node

def is_there_and(node):
    children = node.get_children()
    for child in children:
        if child.get_node_type() == "and":
            return True
    return False

def isCNF(node):
    if(node.get_node_value() == "AND"):
        children = node.get_child_nodes()
        for child_node in children:
            if not (child_node.get_node_value() == "OR"):
                return False
            isCNF(child_node)
    else:
        return not is_there_and(node)
    return True

def isClauses(node):
    if(node.get_node_value() == "OR"):
        for child_node in node.get_child_nodes():
            if not (child_node.get_node_value() == "NOT" | child_node.get_node_type() == "predicate" | child_node.get_node_value() == "function"):
                return False
    else:
        return False
    return True

def isLiteral(node):
    if(node.get_node_value() == "NOT"):
        child_node = node.get_child_nodes()[0]

        if(child_node.get_node_type() == "predicate" | child_node.get_node_type() == "funciton"):
            return True

    if (node.get_node_type() == "function" | node.get_node_type() == "predicate"):
        return True

    return False

def concatenate(node_tuple):
    new_children=[]
    for node in node_tuple:
        new_children.extend(node.get_child_nodes())

    parent = Node("op","OR")
    parent.set_child_nodes(new_children)
    return parent

# make the distribution to derive to the CNF
def convert_to_CNF(node):

    child_nodes = node.get_children()
    no_child_nodes = len(child_nodes)

    if isCNF(node):
        return node

    if isClauses(node):
        new_node = Node("op","AND")
        new_node.set_child_nodes([node])
        return new_node

    if isLiteral(node):
        new_node = Node("op","AND")
        new_parent = Node("op","OR")

        new_parent_node = Node("op", "OR")
        new_parent_node.set_child_nodes([node])

        new_node.set_child_nodes([new_parent])

        return new_node

    if node.get_node_value() == "AND" and no_child_nodes > 0:
        new_children = []
        for i in range(no_child_nodes):
            child_node = child_nodes[i]
            X_tree = convert_to_CNF(child_node)
            x_child_nodes = X_tree.get_child_nodes()
            for x_child_node in x_child_nodes:
                new_children.append(x_child_node)

        new_node = Node("op","AND")
        new_node.set_child_nodes(new_children)
        return new_node

    if node.get_node_value() == "OR" and no_child_nodes > 0:
        new_children = []
        for i in range(no_child_nodes):
            child_node = child_nodes[i]

            X_tree = convert_to_CNF(child_node)
            x_child_nodes = X_tree.get_child_nodes()
            x_children = []
            for x_child_nodes in x_child_nodes:
                x_children.append(x_child_nodes)

        new_combined_children = []
        for tuple in new_combined_children:
            new_x_node = concatenate(tuple)
            new_x_children.append(new_x_node)

        new_node = Node("op","AND")
        new_node.set_child_nodes(new_x_children)

        return new_node

    else:
        print("Error.")
# prenex part which pushing all the quantifiers up as far as possible.
####################################################################
def is_prenex(node):
    children = node.get_children()

    if len(children) != 2:
        return True

    if( node.get_node_type() != "quant" and (children[1].get_node_type() == "quant" or children[0].get_node_type() == "quant")):
        return False
    else:
        return is_prenex(children[0]) and is_prenex(children[1])

def convert_prenex(node):
    children = node.get_children()
    if len(children) != 2:
        return node
    current_node = node
    left_child = children[1]
    left_child_children = left_child.get_children()
    right_child = children[0]
    right_child_children = right_child.get_children()

    if(current_node.get_node_type() == "op" and left_child.get_node_type() == "quant"):
        left_child_left = left_child_children[1]
        leftChildLeftType = left_child_left.get_element_type()
        leftChildLeftValue = left_child_left.get_element_value()

        temp_type = current_node.get_node_type()
        temp_value = current_node.get_node_value()

        current_node.set_node(leftChildType, leftChildValue)
        left_child.set_node(temp_type, temp_value)

        temp = Node(leftChildLeftType, leftChildLeftValue)
        left_child.get_children()[1] = left_child.get_children()[0]
        left_child.get_children()[0] = node.get_children()[0]
        node.get_children()[0] = node.get_children()[1]
        node.get_children()[1] = temp

    elif (current_node.get_node_type() == "op" and right_child.get_node_type() == "quant"):
        rightChildLeft = rightChildChildren[1]
        rightChildLeftType = rightChildLeft.get_node_type()
        rightChildLeftValue = rightChildLeft.get_node_value()

        tempType = current_node.get_node_type()
        tempValue = current_node.get_node_value()

        current_node.set_node(rightChildType, rightChildValue)
        rightChild.set_node(tempType, tempValue)

        temp = Node(rightChildLeftType, rightChildLeftValue)
        rightChild.get_child_nodes()[1] = current_node.get_child_nodes()[1]
        current_node.get_child_nodes()[1] = temp

    convert_prenex(left_child)
    convert_prenex(right_child)
    return node

def prenex_form(node):
    if is_prenex(node):
        return node
    else:
        return prenex_form(convert_prenex(node))


###########################################################################
variable_list = []

# to make a unique variable for each quantifier
def skolemize(node):
    global variable_list
    current_node = node
    node_type = node.get_node_type()
    _symbol_value = node.get_node_value()

    if node_type != "quant":
        variable_list = []
        return node
    else:
        currChildren = current_node.get_child_nodes()

        leftChild = currChildren[1]
        leftChildValue = leftChild.get_node_value()

        rightChild = currChildren[0]
        rightChildType = rightChild.get_node_type()
        rightChildValue = rightChild.get_node_value()

        rightChildChildren = rightChild.get_child_nodes()
        if len(rightChildChildren) == 2:
            rightChildleft = rightChildChildren[1]
        rightChildRight = rightChildChildren[0]
        if _symbol_value == "FORALL":
            varList.append(leftChildValue)

        else:  # EXISTS
            if variable_list == []:
                new_symbol_type = "symbol"
            else:
                new_symbol_type = "function"
            rename(node, new_symbol_type, leftChildValue)

            current_node.set_node(rightChildType, rightChildValue)
            if len(rightChildChildren) == 1:
                currChildren.pop(1)

            for i in range(0, len(rightChildChildren)):
                currChildren[i] = rightChildChildren[i]
            skolemize(current_node)

        skolemize(currChildren[0])
        return node

def rename(node, new_symbol_type, variable):
    global variable_list
    current_node = node
    children = node.get_children()

    if node.get_node_value == variable:
        if new_symbol_type == "symbol":
            current_node.set_node(new_symbol_type, variable)
        else:
            current_node.set_node(new_symbol_type, variable)
            for i in range(0,len(variable_list)):
                temp = Node("variable",variable_list[i])
                current_node.append_child(temp)

    else:
        for i in range(len(children)):
            node = rename(children[i], new_symbol_type, variable)

    return node

universal_variable_list = []


def drop_universal(node):
    global universal_variable_list
    node_type = node.get_node_type()
    currentNode = node
    currChildren = currentNode.get_child_nodes()

    if len(currChildren) != 2:
        return node
    else:
        if node_type == "quant":
            leftChild = currChildren[1]
            leftChildValue = leftChild.get_node_value()
            universal_varList.append(leftChildValue)

            rightChild = currChildren[0]
            rightChildType = rightChild.get_node_type()
            rightChildValue = rightChild.get_node_value()

            rightChildChildren = rightChild.get_children()

            currentNode.set_node(rightChildType, rightChildValue)
            if len(rightChildChildren) == 1:
                currChildren.pop(1)

            for i in range(0, len(rightChildChildren)):
                currChildren[i] = rightChildChildren[i]

            drop_universal(currentNode)

        drop_universal(currChildren[0])
        return node


def symbol_fixer(node):
    currentNode = node

    if currentNode.get_node_type() == "variable":
        if currentNode.get_node_value() not in universal_variable_list:
            currentNode.set_node("symbol", currentNode.get_node_value())

    if len(currentNode.get_children()) == 0:
        return node

    for i in range(0, len(currentNode.get_children())):
        symbol_fixer(currentNode.get_children()[i])

    return node

# some other classes
#############################################################################
VARIABLE = "VARIABLE"
CONSTANT = "CONSTANT"


class Argument(object):
    def __init__(self, name, kind):
        self._name = name
        self._kind = kind

    def is_variable(self):
        return self._kind == VARIABLE

    def is_constant(self):
        return self._kind == CONSTANT

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def same_kind(self, arg):

        return self._kind == arg._kind

    def equals(self, arg):

        return (self.same_kind(arg) and self._name == arg._name)

    @classmethod
    def make_var(cls, name):
        return Argument(name, VARIABLE)

    @classmethod
    def make_const(cls, name):
        return Argument(name, CONSTANT)


class Predicate(object):

    def __init__(self, name, args, negative=False):

        self._name = name
        self._args = args
        self._negative = negative

    def __str__(self):
        s = "{0}({1})".format(self._name, ",".join([arg._name for arg in self._args]))
        if self._negative:
            return "-"+s

        return s

    def get_name(self):
        return self._name

    def get_args(self):
        return self._args

    def get_negative(self):
        return self._negative

    def same_formula(self, obj):

        if self._name != obj._name:
            return False

        if len(self._args) != len(obj._args):
            return False

        return True

    def complement_of(self, obj):

        return (self.same_formula(obj) and self._negative != obj._negative)

    def same_args(self, obj):

        for i in range(0, len(obj._args)):
            if not self._args[i].equals(obj._args[i]):
                return False

        return True

    def equals(self, obj):

        if not self.same_formula(obj):
            return False

        if not self.same_args(obj):
            return False

        if self._negative != obj._negative:
            return False

        return True

    def same_predicate(self, obj):

        if self._name != obj._name:
            return False

        return True

    def complement_of_predicate(self, obj):

        if self.same_predicate(obj) == True and self._negative != obj._negative:
            return True

        return False

##########################################################################################
def unification(p1, p2, replacements):

    p1_args = list(p1.get_args())
    p2_args = list(p2.get_args())

    if len(p1_args) != len(p2_args):
        return p1, p2, False

    if p1.same_args(p2):
        return p1, p2, True

    for i in range(0, len(p1_args)):
        p1_arg = p1_args[i]
        p2_arg = p2_args[i]

        if p2_arg.equals(p1_arg):
            continue

        if p1_arg.is_variable() and p2_arg.is_variable():
            token = replacements.get(p2_arg.get_name(), '')
            if token == '':
                token = p1_arg.get_name()
                replacements[p2_arg.get_name()] = token

            p1_args[i].set_name(token)
            p2_args[i].set_name(token)

            continue

        const = ''
        var = ''
        if p1_arg.is_constant() and p2_arg.is_variable():
            const = p1_arg.get_name()
            var = p2_arg.get_name()
        else:
            const = p2_arg.get_name()
            var = p1_arg.get_name()

        if '({0})'.format(const) in var: # can't to unification
            return p1, p2, False

        replacements[var] = const
        p1_args[i].set_name(const)
        p2_args[i].set_name(const)

    p1._args = p1_args
    p2._args = p2_args

    return p1, p2, True

# sort the clauses by length and for each clause trying to resolve it
# with the remaining clauses and add new clauses if made
# returns true if it's derived an empty clause and false otherwise
def resolution(clauses):
    if len(clauses) == 1:
        return False
    setofSupport = list(clauses)
    second_set = list(clauses)
    intinal_size = len(second_set)
    clauses.sort(key=len)
    for (x, y) in enumerate(clauses):
        current_clause = clauses[x]
        setofSupport.remove(current_clause)
        newClause = addNewClause(current_clause,setofSupport)
        if newClause ==None:
            return False
        elif len(newClause) ==0:
            return True
        else:
            if newClause not in second_set:
                second_set.append(setofSupport)

    new_size = len(second_set)
    if new_size > intinal_size:
        resolution(second_set)
    return False

def addNewClause(refSet,setofSupport):
    newClause =[]
    for (i, e) in enumerate(refSet):
        setofSupport.sort(key=len)
        for (j,k) in enumerate(setofSupport):
            check = False
            for(m,n) in enumerate(k):
                if  e.complement_of_predicate(n) or check == True:
                    check = True
                    reps = dict()
                    p1, p2, flag = unification(e, n, reps)
                    if flag == True:
                        n = p2
                        e = p1
                    else:
                        continue
            for (a,b) in enumerate(k):
                if e.complement_of(b):
                    newClause = (setofSupport[j])
                    newClause.remove(b)
                    for (c,d) in enumerate(refSet):
                        if d==e:
                            continue
                        elif d in newClause:
                            continue
                        else:
                            newClause.append(d)
                    return newClause
                else:
                    continue

    return None

# takes a list of nodes and return a list of arguments
def get_args_from_nodes(nodes):
    args = list()
    for node in nodes:
        if node.get_element_type() == 'function':
            symbols = node.get_child_nodes()
            arg_token = ','.join([x.get_element_value() for x in symbols])
            const = '{0}({1})'.format(node.get_element_value(), arg_token)
            args.append(Argument.make_const(const))
            continue

        if node.get_element_type() == 'symbol':
            val = node.get_element_value()
            args.append(Argument.make_const(val))
            continue

        if node.get_element_type() == 'variable':
            val = node.get_element_value()
            args.append(Argument.make_var(val))
            continue

    return args

# convert the whole CNF into clauses
def CNF_to_clausal(and_node):

    or_list = and_node.get_child_nodes()
    clauses = list()
    for r in or_list:
        for node in r.get_child_nodes():
            # there's only one predicate here, get that
            if node.get_element_type() == 'op' and node.get_element_value() == 'NOT':
                predicate = node.get_child_nodes()[0]
                args = get_args_from_nodes(predicate.get_child_nodes())
                p = Predicate(predicate.get_element_value(), args, True)
                clauses.append(p)
            else:
                args = get_args_from_nodes(node.get_child_nodes())
                p = Predicate(node.get_element_value(), args)
                clauses.append(p)

    return clauses


####################################################################################################
def solve(KB,alpha):
    ALL_statments = [KB,alpha]
    ALL_clauses = []
    for statment in ALL_statments:
        if statment == ALL_statments[-1]:
            add_not = True
        else:
            add_not = False
        tree = parse(statment)
        if add_not:
            negation_node = Node("op","NOT")
            negation_node.append_child(tree)
            tree = negation_node
        tree = implication_elimination(tree)
        tree = deMorgan_mvNegation(tree)
        tree = standardize(tree)
        tree = prenex_form(tree)
        tree = skolemize(tree)
        tree = drop_universal(tree)
        tree = convert_to_CNF(tree)
        clauses = CNF_to_clauses(tree)
        ALL_clauses.append(clauses)

    return resolution(ALL_clauses)



##########################################################################




