import re
from typing import List, Union

# Definir operadores válidos
COMPARISON_OPS = {'=': '==', '<': '<', '≤': '<=', '>': '>', '≥': '>='}
BOOL_OPS = {'∨': 'or', '∧': 'and'}

# Expresión regular mejorada para capturar restricciones
CONSTRAINT_PATTERN = re.compile(r"([a-zA-Z0-9+\-*\/^\s]+)\s*(=|<|≤|>|≥)\s*([a-zA-Z0-9+\-*\/^\s]+)")

class ConstraintParser:
    def __init__(self, constraint_str: str):
        self.constraint_str = constraint_str

    def parse(self) -> str:
        """Convierte la expresión de restricciones en código Python evaluable."""
        expression = self.constraint_str
        
        # Reemplazar operadores de comparación
        for op, py_op in COMPARISON_OPS.items():
            expression = expression.replace(op, f' {py_op} ')
        
        # Reemplazar operadores booleanos
        for op, py_op in BOOL_OPS.items():
            expression = expression.replace(op, f' {py_op} ')
        
        return expression

    def extract_constraints(self) -> List[str]:
        """Extrae restricciones individuales de la expresión."""
        return [match.group(0) for match in CONSTRAINT_PATTERN.finditer(self.constraint_str)]
    

if __name__ == "__main__":

    # Ejemplo de uso
    constraint_string = "(x1 > 1 ∨ x2 < 2) ∧ (x1 + x2 ≤ 10)"
    parser = ConstraintParser(constraint_string)
    parsed_constraints = parser.parse()
    extracted = parser.extract_constraints()

    print("Expresión traducida a Python:", parsed_constraints)
    print("Restricciones individuales:", extracted)