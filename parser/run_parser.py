from keyword import iskeyword
from parser import (
    DFG_c,
    DFG_python,
    DFG_java,
    DFG_ruby,
    DFG_go,
    DFG_php,
    DFG_javascript,
    DFG_csharp,
)
from parser import (
    remove_comments_and_docstrings,
    tree_to_token_index,
    index_to_code_token,
    tree_to_variable_index,
)
from tree_sitter import Language, Parser

dfg_function = {
    "c": DFG_c,
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
    "csharp": DFG_csharp,
}

parsers = {}
for lang in dfg_function:
    LANGUAGE = Language("../parser./my-languages.so", lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser

python_keywords = [
    "import",
    "",
    "[",
    "]",
    ":",
    ",",
    ".",
    "(",
    ")",
    "{",
    "}",
    "not",
    "is",
    "=",
    "+=",
    "-=",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
]
java_keywords = [
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
]
java_special_ids = [
    "main",
    "args",
    "Math",
    "System",
    "Random",
    "Byte",
    "Short",
    "Integer",
    "Long",
    "Float",
    "Double",
    "Character",
    "Boolean",
    "Data",
    "ParseException",
    "SimpleDateFormat",
    "Calendar",
    "Object",
    "String",
    "StringBuffer",
    "StringBuilder",
    "DateFormat",
    "Collection",
    "List",
    "Map",
    "Set",
    "Queue",
    "ArrayList",
    "HashSet",
    "HashMap",
]
c_keywords = [
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Alignas",
    "_Alignof",
    "_Atomic",
    "_Bool",
    "_Complex",
    "_Generic",
    "_Imaginary",
    "_Noreturn",
    "_Static_assert",
    "_Thread_local",
    "__func__",
]

c_macros = [
    "NULL",
    "_IOFBF",
    "_IOLBF",
    "BUFSIZ",
    "EOF",
    "FOPEN_MAX",
    "TMP_MAX",  # <stdio.h> macro
    "FILENAME_MAX",
    "L_tmpnam",
    "SEEK_CUR",
    "SEEK_END",
    "SEEK_SET",
    "NULL",
    "EXIT_FAILURE",
    "EXIT_SUCCESS",
    "RAND_MAX",
    "MB_CUR_MAX",
]  # <stdlib.h> macro
c_special_ids = [
    "main",  # main function
    "stdio",
    "cstdio",
    "stdio.h",  # <stdio.h> & <cstdio>
    "size_t",
    "FILE",
    "fpos_t",
    "stdin",
    "stdout",
    "stderr",  # <stdio.h> types & streams
    "remove",
    "rename",
    "tmpfile",
    "tmpnam",
    "fclose",
    "fflush",  # <stdio.h> functions
    "fopen",
    "freopen",
    "setbuf",
    "setvbuf",
    "fprintf",
    "fscanf",
    "printf",
    "scanf",
    "snprintf",
    "sprintf",
    "sscanf",
    "vprintf",
    "vscanf",
    "vsnprintf",
    "vsprintf",
    "vsscanf",
    "fgetc",
    "fgets",
    "fputc",
    "getc",
    "getchar",
    "putc",
    "putchar",
    "puts",
    "ungetc",
    "fread",
    "fwrite",
    "fgetpos",
    "fseek",
    "fsetpos",
    "ftell",
    "rewind",
    "clearerr",
    "feof",
    "ferror",
    "perror",
    "getline" "stdlib",
    "cstdlib",
    "stdlib.h",  # <stdlib.h> & <cstdlib>
    "size_t",
    "div_t",
    "ldiv_t",
    "lldiv_t",  # <stdlib.h> types
    "atof",
    "atoi",
    "atol",
    "atoll",
    "strtod",
    "strtof",
    "strtold",  # <stdlib.h> functions
    "strtol",
    "strtoll",
    "strtoul",
    "strtoull",
    "rand",
    "srand",
    "aligned_alloc",
    "calloc",
    "malloc",
    "realloc",
    "free",
    "abort",
    "atexit",
    "exit",
    "at_quick_exit",
    "_Exit",
    "getenv",
    "quick_exit",
    "system",
    "bsearch",
    "qsort",
    "abs",
    "labs",
    "llabs",
    "div",
    "ldiv",
    "lldiv",
    "mblen",
    "mbtowc",
    "wctomb",
    "mbstowcs",
    "wcstombs",
    "string",
    "cstring",
    "string.h",  # <string.h> & <cstring>
    "memcpy",
    "memmove",
    "memchr",
    "memcmp",
    "memset",
    "strcat",  # <string.h> functions
    "strncat",
    "strchr",
    "strrchr",
    "strcmp",
    "strncmp",
    "strcoll",
    "strcpy",
    "strncpy",
    "strerror",
    "strlen",
    "strspn",
    "strcspn",
    "strpbrk",
    "strstr",
    "strtok",
    "strxfrm",
    "memccpy",
    "mempcpy",
    "strcat_s",
    "strcpy_s",
    "strdup",  # <string.h> extension functions
    "strerror_r",
    "strlcat",
    "strlcpy",
    "strsignal",
    "strtok_r",
    "iostream",
    "istream",
    "ostream",
    "fstream",
    "sstream",  # <iostream> family
    "iomanip",
    "iosfwd",
    "ios",
    "wios",
    "streamoff",
    "streampos",
    "wstreampos",  # <iostream> types
    "streamsize",
    "cout",
    "cerr",
    "clog",
    "cin",
    "boolalpha",
    "noboolalpha",
    "skipws",
    "noskipws",
    "showbase",  # <iostream> manipulators
    "noshowbase",
    "showpoint",
    "noshowpoint",
    "showpos",
    "noshowpos",
    "unitbuf",
    "nounitbuf",
    "uppercase",
    "nouppercase",
    "left",
    "right",
    "internal",
    "dec",
    "oct",
    "hex",
    "fixed",
    "scientific",
    "hexfloat",
    "defaultfloat",
    "width",
    "fill",
    "precision",
    "endl",
    "ends",
    "flush",
    "ws",
    "showpoint",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",  # <math.h> functions
    "cosh",
    "tanh",
    "exp",
    "sqrt",
    "log",
    "log10",
    "pow",
    "powf",
    "ceil",
    "floor",
    "abs",
    "fabs",
    "cabs",
    "frexp",
    "ldexp",
    "modf",
    "fmod",
    "hypot",
    "ldexp",
    "poly",
    "matherr",
]

go_keywords = [
    "break",
    "default",
    "func",
    "interface",
    "select",
    "case",
    "defer",
    "go",
    "map",
    "struct",
    "chan",
    "else",
    "goto",
    "package",
    "switch",
    "const",
    "fallthrough",
    "if",
    "range",
    "type",
    "continue",
    "for",
    "import",
    "return",
    "var",
]

go_predeclared_types = [
    "bool",
    "byte",
    "complex64",
    "complex128",
    "error",
    "float32",
    "float64",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "rune",
    "string",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uintptr",
]

go_predeclared_constants = ["true", "false", "iota", "nil"]

go_zero_value_identifiers = ["nil"]

go_predeclared_functions = [
    "append",
    "cap",
    "close",
    "complex",
    "copy",
    "delete",
    "imag",
    "len",
    "make",
    "new",
    "panic",
    "print",
    "println",
    "real",
    "recover",
]

js_keywords = [
    "abstract",
    "arguments",
    "await",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "debugger",
    "default",
    "delete",
    "do",
    "double",
    "else",
    "enum",
    "eval",
    "export",
    "extends",
    "false",
    "final",
    "finally",
    "float",
    "for",
    "function",
    "goto",
    "if",
    "implements",
    "import",
    "in",
    "instanceof",
    "int",
    "interface",
    "let",
    "long",
    "native",
    "new",
    "null",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "true",
    "try",
    "typeof",
    "var",
    "void",
    "volatile",
    "while",
    "with",
    "yield",
]


js_fundamental_objects = [
    "Object",
    "Function",
    "Boolean",
    "Symbol",
    "Error",
    "EvalError",
    "InternalError",
    "RangeError",
    "ReferenceError",
    "SyntaxError",
    "TypeError",
    "URIError",
]

js_numbers_dates = ["Number", "Math", "Date"]

js_text_processing = ["String", "RegExp"]

js_indexed_collections = [
    "Array",
    "Int8Array",
    "Uint8Array",
    "Uint8ClampedArray",
    "Int16Array",
    "Uint16Array",
    "Int32Array",
    "Uint32Array",
    "Float32Array",
    "Float64Array",
]

js_keyed_collections = ["Map", "Set", "WeakMap", "WeakSet"]

js_structured_data = ["JSON"]

js_control_abstraction_objects = [
    "Promise",
    "Generator",
    "GeneratorFunction",
    "AsyncFunction",
]

js_reflection = ["Reflect", "Proxy"]

js_global_functions = [
    "eval",
    "isFinite",
    "isNaN",
    "parseFloat",
    "parseInt",
    "decodeURI",
    "decodeURIComponent",
    "encodeURI",
    "encodeURIComponent",
]

php_keywords = [
    "__halt_compiler",
    "abstract",
    "and",
    "array",
    "as",
    "break",
    "callable",
    "case",
    "catch",
    "class",
    "clone",
    "const",
    "continue",
    "declare",
    "default",
    "die",
    "do",
    "echo",
    "else",
    "elseif",
    "empty",
    "enddeclare",
    "endfor",
    "endforeach",
    "endif",
    "endswitch",
    "endwhile",
    "eval",
    "exit",
    "extends",
    "final",
    "finally",
    "fn",
    "for",
    "foreach",
    "function",
    "global",
    "goto",
    "if",
    "implements",
    "include",
    "include_once",
    "instanceof",
    "insteadof",
    "interface",
    "isset",
    "list",
    "namespace",
    "new",
    "or",
    "print",
    "private",
    "protected",
    "public",
    "require",
    "require_once",
    "return",
    "static",
    "switch",
    "throw",
    "trait",
    "try",
    "unset",
    "use",
    "var",
    "while",
    "xor",
    "yield",
]

php_string_functions = [
    "addslashes",
    "chop",
    "explode",
    "htmlspecialchars",
    "strlen",
    "strpos",
    "str_replace",
    "str_split",
    "strtolower",
    "strtoupper",
    "trim",
]


php_array_functions = [
    "array_combine",
    "array_count_values",
    "array_diff",
    "array_filter",
    "array_map",
    "array_merge",
    "array_pop",
    "array_push",
    "array_slice",
    "array_unique",
    "sort",
]

php_file_system_functions = [
    "fclose",
    "feof",
    "fgets",
    "fopen",
    "fread",
    "fwrite",
    "glob",
    "is_dir",
    "is_file",
    "mkdir",
    "rename",
    "rmdir",
    "unlink",
]

php_date_time_functions = [
    "date",
    "date_create",
    "date_diff",
    "date_format",
    "getdate",
    "strtotime",
]

php_miscellaneous_functions = [
    "die",
    "empty",
    "eval",
    "exit",
    "isset",
    "print_r",
    "var_dump",
]

php_spl_classes = [
    "ArrayObject",
    "DirectoryIterator",
    "Exception",
    "FileNotFoundException",
    "GlobIterator",
    "InvalidArgumentException",
    "IteratorAggregate",
    "PDOException",
    "RuntimeException",
    "SplDoublyLinkedList",
    "SplFileInfo",
    "SplFileObject",
    "SplFixedArray",
    "SplHeap",
    "SplMinHeap",
    "SplMaxHeap",
    "SplObjectStorage",
    "SplPriorityQueue",
    "SplQueue",
    "SplStack",
    "SplTempFileObject",
]


ruby_keywords = [
    "__ENCODING__",
    "__LINE__",
    "__FILE__",
    "BEGIN",
    "END",
    "alias",
    "and",
    "begin",
    "break",
    "case",
    "class",
    "def",
    "defined?",
    "do",
    "else",
    "elsif",
    "end",
    "ensure",
    "false",
    "for",
    "if",
    "in",
    "module",
    "next",
    "nil",
    "not",
    "or",
    "redo",
    "rescue",
    "retry",
    "return",
    "self",
    "super",
    "then",
    "true",
    "undef",
    "unless",
    "until",
    "when",
    "while",
    "yield",
]

ruby_basic_classes = [
    "Array",
    "Hash",
    "String",
    "Symbol",
    "Numeric",
    "Integer",
    "Float",
    "Range",
    "Regexp",
    "Proc",
    "Lambda",
    "Thread",
    "Exception",
    "StandardError",
    "SystemExit",
    "IO",
    "File",
]


ruby_enumerable_methods = [
    "each",
    "map",
    "select",
    "reject",
    "grep",
    "find",
    "find_all",
    "reduce",
    "any?",
    "all?",
    "none?",
    "count",
    "sort",
    "sort_by",
]


ruby_numeric_classes = [
    "Numeric",
    "Integer",
    "Fixnum",
    "Bignum",
    "Float",
    "Rational",
    "Complex",
]

ruby_math_methods = ["sqrt", "log", "sin", "cos", "tan"]


ruby_io_classes = ["IO", "File", "Dir", "StringIO"]

ruby_io_methods = ["puts", "gets", "print", "printf", "read", "write", "open", "close"]


ruby_networking_classes = ["Socket", "TCPSocket", "TCPServer", "UDPSocket"]

csharp_keywords = [
    "bool",
    "byte",
    "sbyte",
    "char",
    "decimal",
    "double",
    "float",
    "int",
    "uint",
    "long",
    "ulong",
    "object",
    "dynamic",
    "short",
    "ushort",
    "string",
    "void",
    "as",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "default",
    "delegate",
    "do",
    "else",
    "enum",
    "event",
    "explicit",
    "extern",
    "false",
    "finally",
    "fixed",
    "for",
    "foreach",
    "goto",
    "if",
    "implicit",
    "in",
    "interface",
    "internal",
    "is",
    "lock",
    "namespace",
    "new",
    "null",
    "operator",
    "out",
    "override",
    "params",
    "private",
    "protected",
    "public",
    "readonly",
    "ref",
    "return",
    "sealed",
    "sizeof",
    "stackalloc",
    "static",
    "struct",
    "switch",
    "this",
    "throw",
    "true",
    "try",
    "typeof",
    "unchecked",
    "unsafe",
    "using",
    "virtual",
    "volatile",
    "while",
    "abstract",
    "async",
    "await",
    "checked",
    "partial",
    "yield",
]

system_classes = ["Console", "Math", "DateTime", "Guid", "Random", "Convert"]

collections_generic = [
    "List<T>",
    "Dictionary<TKey, TValue>",
    "HashSet<T>",
    "Queue<T>",
    "Stack<T>",
]

system_io_classes = [
    "File",
    "FileInfo",
    "Directory",
    "DirectoryInfo",
    "FileStream",
    "StreamReader",
    "StreamWriter",
]

system_text_classes = ["StringBuilder", "Encoding"]

system_net_classes = ["WebClient", "HttpWebRequest", "HttpWebResponse", "Socket"]

system_threading_classes = ["Thread", "ThreadPool", "Mutex", "Semaphore"]

system_threading_tasks = ["Task", "Task<T>"]

system_linq_classes = ["Enumerable", "Queryable"]


special_char = [
    "[",
    "]",
    ":",
    ",",
    ".",
    "(",
    ")",
    "{",
    "}",
    "not",
    "is",
    "=",
    "+=",
    "-=",
    "<",
    ">",
    "+",
    "-",
    "*",
    "/",
    "|",
]


def is_valid_variable_python(name: str) -> bool:
    return name.isidentifier() and not iskeyword(name)


def is_valid_variable_java(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in java_keywords:
        return False
    elif name in java_special_ids:
        return False
    return True


def is_valid_variable_c(name: str) -> bool:

    if not name.isidentifier():
        return False
    elif name in c_keywords:
        return False
    elif name in c_macros:
        return False
    elif name in c_special_ids:
        return False
    return True


def is_valid_variable_go(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in go_keywords:
        return False
    elif name in go_predeclared_types:
        return False
    elif name in go_predeclared_constants:
        return False
    elif name in go_zero_value_identifiers:
        return False
    elif name in go_predeclared_functions:
        return False
    return True


def is_valid_variable_js(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in js_keywords:
        return False
    elif name in js_fundamental_objects:
        return False
    elif name in js_numbers_dates:
        return False
    elif name in js_text_processing:
        return False
    elif name in js_indexed_collections:
        return False
    elif name in js_keyed_collections:
        return False
    elif name in js_structured_data:
        return False
    elif name in js_control_abstraction_objects:
        return False
    elif name in js_reflection:
        return False
    elif name in js_global_functions:
        return False
    return True


def is_valid_variable_php(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in php_keywords:
        return False
    elif name in php_string_functions:
        return False
    elif name in php_array_functions:
        return False
    elif name in php_file_system_functions:
        return False
    elif name in php_date_time_functions:
        return False
    elif name in php_miscellaneous_functions:
        return False
    elif name in php_spl_classes:
        return False
    return True


def is_valid_variable_ruby(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in ruby_keywords:
        return False
    elif name in ruby_basic_classes:
        return False
    elif name in ruby_enumerable_methods:
        return False
    elif name in ruby_numeric_classes:
        return False
    elif name in ruby_math_methods:
        return False
    elif name in ruby_io_classes:
        return False
    elif name in ruby_io_methods:
        return False
    elif name in ruby_networking_classes:
        return False
    return True


def is_valid_variable_csharp(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in csharp_keywords:
        return False
    elif name in system_classes:
        return False
    elif name in collections_generic:
        return False
    elif name in system_io_classes:
        return False
    elif name in system_text_classes:
        return False
    elif name in system_net_classes:
        return False
    elif name in system_threading_classes:
        return False
    elif name in system_threading_tasks:
        return False
    elif name in system_linq_classes:
        return False
    return True


def is_valid_variable_name(name: str, lang: str) -> bool:
    # check if matches language keywords
    if lang == "python":
        return is_valid_variable_python(name)
    elif lang == "c":
        return is_valid_variable_c(name)
    elif lang == "java":
        return is_valid_variable_java(name)
    elif lang == "go":
        return is_valid_variable_go(name)
    elif lang == "javascript":
        return is_valid_variable_js(name)
    elif lang == "php":
        return is_valid_variable_php(name)
    elif lang == "ruby":
        return is_valid_variable_ruby(name)
    elif lang == "csharp":
        return is_valid_variable_csharp(name)
    else:
        return False


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def get_identifiers(code, lang):

    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    ret = []
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.append(d[0])
    ret = unique(ret)
    ret = [[i] for i in ret]
    return ret, code_tokens


def extract_dataflow(code, lang):
    parser = parsers[lang]
    code = code.replace("\\n", "\n")
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    tree = parser[0].parse(bytes(code, "utf8"))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split("\n")
    # print(code)
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)

    index_table = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_table[idx] = index

    DFG, _ = parser[1](root_node, index_to_code, {})

    DFG = sorted(DFG, key=lambda x: x[1])
    return DFG, index_table, code_tokens
