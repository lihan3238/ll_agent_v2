# src/tools/code_utils.py
import ast
import os
from src.core.schema import FileSpec

class CodeUtils:
    @staticmethod
    def generate_skeleton_from_design(file_spec: FileSpec) -> str:
        lines = []
        clean_filename = file_spec.filename.replace("\\", "/")
        
        # 1. Imports
        if file_spec.imports:
            lines.extend(file_spec.imports)
            lines.append("")

        # [删除] 全局 raise，这会阻止文件被 import
        # lines.append("raise NotImplementedError(...)") 

        # 2. Classes
        if file_spec.classes:
            for cls in file_spec.classes:
                inherits = cls.inherits_from if cls.inherits_from else "object"
                lines.append(f"class {cls.name}({inherits}):")
                lines.append(f'    """{cls.description}"""')
                
                if cls.attributes:
                    for attr in cls.attributes:
                        lines.append(f"    # {attr}")
                lines.append("")

                if cls.methods:
                    for method in cls.methods:
                        args = method.args
                        args_str = ", ".join(args) if isinstance(args, list) else str(args)
                        lines.append(f"    def {method.name}({args_str}) -> {method.return_type}:")
                        lines.append(f'        """{method.docstring}"""')
                        # 在函数体内抛出异常是合理的，逼迫 AI 去覆盖
                        lines.append("        raise NotImplementedError('Method not implemented')") 
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
                lines.append("    raise NotImplementedError('Function not implemented')")
                lines.append("")

        # 4. Main Guard
        if "main.py" in clean_filename:
            lines.append('if __name__ == "__main__":')
            lines.append("    raise NotImplementedError('Main execution not implemented')")

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