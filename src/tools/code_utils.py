# src/tools/code_utils.py
import ast
import os
from src.core.schema import FileSpec

class CodeUtils:
    @staticmethod
    def generate_skeleton_from_design(file_spec: FileSpec) -> str:
        """
        [确定性生成] 根据 Architect 的设计，直接生成 Python 骨架代码。
        """
        lines = []
        
        # [优化] 路径清洗：确保内部处理时路径是 POSIX 风格
        # 虽然写入时也会洗，但这里洗一下能保证日志好看
        clean_filename = file_spec.filename.replace("\\", "/")
        
        # 1. Imports
        if file_spec.imports:
            lines.extend(file_spec.imports)
            lines.append("") # 空行

        # 2. Classes
        if file_spec.classes:
            for cls in file_spec.classes:
                # Class Definition
                inherits = cls.inherits_from if cls.inherits_from else "object"
                lines.append(f"class {cls.name}({inherits}):")
                lines.append(f'    """{cls.description}"""')
                
                if cls.attributes:
                    for attr in cls.attributes:
                        lines.append(f"    # {attr}")
                lines.append("")

                # Methods
                if cls.methods:
                    for method in cls.methods:
                        # 参数处理
                        args = method.args
                        args_str = ", ".join(args) if isinstance(args, list) else str(args)
                        if "self" not in args_str and "classmethod" not in args_str and "staticmethod" not in args_str:
                            pass 

                        lines.append(f"    def {method.name}({args_str}) -> {method.return_type}:")
                        lines.append(f'        """{method.docstring}"""')
                        lines.append("        pass") # 骨架占位
                        lines.append("")
                else:
                    lines.append("    pass")
                lines.append("")

        # 3. Global Functions
        if file_spec.functions:
            for func in file_spec.functions:
                args_str = ", ".join(func.args) if isinstance(func.args, list) else str(func.args)
                lines.append(f"def {func.name}({args_str}) -> {func.return_type}:")
                lines.append(f'    """{func.docstring}"""')
                lines.append("    pass")
                lines.append("")

        # 4. Main Guard
        if "main.py" in clean_filename:
            lines.append('if __name__ == "__main__":')
            lines.append('    pass')

        return "\n".join(lines)

    @staticmethod
    def extract_ast_skeleton(file_path: str) -> str:
        """
        [AST 压缩] 读取一个 Python 文件，剔除函数体，保留签名和 Imports。
        """
        if not os.path.exists(file_path):
            return f"# File not found: {file_path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            class SkeletonVisitor(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    # 保留 Docstring
                    new_body = []
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str)):
                        new_body.append(node.body[0]) 
                    
                    # 替换其余部分为 ...
                    new_body.append(ast.Expr(value=ast.Constant(value=...)))
                    node.body = new_body
                    return node

                def visit_ClassDef(self, node):
                    self.generic_visit(node)
                    return node
                
                # [新增] 保留 Import 节点，这对理解依赖很有用
                def visit_Import(self, node):
                    return node
                def visit_ImportFrom(self, node):
                    return node

            transformer = SkeletonVisitor()
            new_tree = transformer.visit(tree)
            
            try:
                return ast.unparse(new_tree)
            except AttributeError:
                # Python < 3.9 兼容
                return source 

        except Exception as e:
            # 如果解析失败（比如语法错误），返回原文的前 500 行作为降级
            return f"# AST Parse Error: {e}\n# Raw Content Start:\n{source[:2000]}..."

code_utils = CodeUtils()