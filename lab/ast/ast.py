from clang import cindex
cpp_file_path = "./hello_world.cpp"


# Function to recursively traverse the AST and print information about each node
def traverse(node, depth=0):
    print('  ' * depth, end='')
    print(f'{node.kind.name} ({node.spelling})')
    ast_file.write('  ' * depth)
    ast_file.write(f'{node.kind.name} ({node.spelling})\n')

    for child in node.get_children():
        traverse(child, depth + 1)

def rewrite_ast(node):
    if node.kind == cindex.CursorKind.FUNCTION_DECL and node.spelling == "main":
        print("int main() {")
    elif node.kind == cindex.CursorKind.CALL_EXPR and node.spelling == "operator<<":
        print("std::cout << ", end="")
    elif node.kind == cindex.CursorKind.UNEXPOSED_EXPR:
        if "STRING_LITERAL" in node.spelling:
            print(f'"{node.spelling}"', end="")
        elif node.spelling == "endl":
            print("std::endl", end="")
        elif node.spelling == "operator<<":
            print(" << ", end="")
    # else:
    for child in node.get_children():
        rewrite_ast(child)


if __name__ == "__main__":
    ast_file = open('ast_file.txt','a')
    cindex.Config.set_library_file('/usr/lib/llvm-14/lib/libclang-14.so.1')  # 声明libclang的位置
    index = cindex.Index.create()

    # Parse the translation unit
    translation_unit = index.parse(cpp_file_path)

    # if translation_unit and not translation_unit.diagnostics:
    #     print('Parsing successful...')

    #     # rewriter = cindex.Rewriter.from_source_file(cpp_file_path)
    #     rewrite_ast(translation_unit.cursor)
    #     # Output the modified source code
    #     # modified_code = str(rewriter)
    #     # print("Modified Source Code:")
    #     # print(modified_code)
    # else:
    #     for diag in translation_unit.diagnostics:
    #         print(f"Error: {diag.severity.name} - {diag.spelling}")

    traverse(translation_unit.cursor)