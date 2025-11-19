import math

# --- helper functions ---

def safe_eval(func_str, x):
    """
    Evaluates a mathematical string function f(x) at a given x.
    Allowed context: all functions in math module (sin, cos, exp, etc.).
    """
    allowed_locals = math.__dict__
    allowed_locals['x'] = x
    try:
        return eval(func_str, {"__builtins__": {}}, allowed_locals)
    except Exception as e:
        return None

def derive_eval(deriv_str, x):
    """Helper for Newton-Raphson derivative evaluation."""
    return safe_eval(deriv_str, x)

def print_header(method_name):
    print(f"\n{'='*60}")
    print(f"{method_name.upper()} METHOD")
    print(f"{'='*60}")

def print_row(iter_num, x_val, f_val, error):
    print(f"{iter_num:<10} | {x_val:<18.8f} | {f_val:<18.8f} | {error:<18.8f}")

# --- SOLVER IMPLEMENTATIONS ---

def bisection(func_str, a, b, tol, max_iter):
    print_header("Bisection")
    
    fa = safe_eval(func_str, a)
    fb = safe_eval(func_str, b)
    
    if fa is None or fb is None:
        print("Error: Could not evaluate function.")
        return
    
    if fa * fb >= 0:
        print("Error: f(a) and f(b) must have opposite signs.")
        return

    print(f"{'Iter':<10} | {'Root Est (c)':<18} | {'f(c)':<18} | {'Error':<18}")
    print("-" * 70)

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = safe_eval(func_str, c)
        error = abs(b - a)
        
        print_row(i, c, fc, error)
        
        if abs(fc) < tol or error < tol:
            print(f"\nConverged to root: {c:.8f} after {i} iterations.")
            return

        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc
            
    print("\nMax iterations reached.")

def regula_falsi(func_str, a, b, tol, max_iter):
    print_header("Regula Falsi (False Position)")
    
    fa = safe_eval(func_str, a)
    fb = safe_eval(func_str, b)
    
    if fa * fb >= 0:
        print("Error: f(a) and f(b) must have opposite signs.")
        return

    print(f"{'Iter':<10} | {'Root Est (c)':<18} | {'f(c)':<18} | {'Error':<18}")
    print("-" * 70)

    prev_c = a
    for i in range(1, max_iter + 1):
        # Formula: c = (a*f(b) - b*f(a)) / (f(b) - f(a))
        curr_c = (a * fb - b * fa) / (fb - fa)
        fc = safe_eval(func_str, curr_c)
        
        error = abs(curr_c - prev_c)
        print_row(i, curr_c, fc, error)
        
        if abs(fc) < tol or error < tol:
            print(f"\nConverged to root: {curr_c:.8f} after {i} iterations.")
            return

        if fa * fc < 0:
            b = curr_c
            fb = fc
        else:
            a = curr_c
            fa = fc
        prev_c = curr_c
            
    print("\nMax iterations reached.")

def secant(func_str, x0, x1, tol, max_iter):
    print_header("Secant")
    print(f"{'Iter':<10} | {'Root Est (x2)':<18} | {'f(x2)':<18} | {'Error':<18}")
    print("-" * 70)

    for i in range(1, max_iter + 1):
        f0 = safe_eval(func_str, x0)
        f1 = safe_eval(func_str, x1)
        
        if f1 - f0 == 0:
            print("Error: Division by zero (f(x1) - f(x0) = 0).")
            return

        x2 = x1 - (f1 * (x1 - x0)) / (f1 - f0)
        f2 = safe_eval(func_str, x2)
        error = abs(x2 - x1)
        
        print_row(i, x2, f2, error)
        
        if abs(f2) < tol or error < tol:
            print(f"\nConverged to root: {x2:.8f} after {i} iterations.")
            return
        
        x0, x1 = x1, x2

    print("\nMax iterations reached.")

def newton_raphson(func_str, deriv_str, x0, tol, max_iter):
    print_header("Newton-Raphson")
    print(f"{'Iter':<10} | {'Root Est (x1)':<18} | {'f(x1)':<18} | {'Error':<18}")
    print("-" * 70)

    for i in range(1, max_iter + 1):
        f0 = safe_eval(func_str, x0)
        f_prime0 = derive_eval(deriv_str, x0)
        
        if f_prime0 == 0:
            print("Error: Derivative is zero. Method fails.")
            return
        
        x1 = x0 - (f0 / f_prime0)
        f1 = safe_eval(func_str, x1)
        error = abs(x1 - x0)
        
        print_row(i, x1, f1, error)
        
        if abs(f1) < tol or error < tol:
            print(f"\nConverged to root: {x1:.8f} after {i} iterations.")
            return
        
        x0 = x1

    print("\nMax iterations reached.")

def fixed_point(g_str, x0, tol, max_iter):
    # Note: User enters g(x) where f(x) = x - g(x) = 0 OR x = g(x)
    print_header("Fixed Point Iteration")
    print(f"Solving for x = g(x) using g(x) = {g_str}")
    print(f"{'Iter':<10} | {'Root Est (x1)':<18} | {'g(x1)':<18} | {'Error':<18}")
    print("-" * 70)

    for i in range(1, max_iter + 1):
        x1 = safe_eval(g_str, x0)
        if x1 is None: return
        
        error = abs(x1 - x0)
        
        print_row(i, x1, x1, error) # g(x1) is technically the next x
        
        if error < tol:
            print(f"\nConverged to root: {x1:.8f} after {i} iterations.")
            return
        
        x0 = x1
        
        if x0 > 1e10: # Divergence check
            print("\nValues getting too large. Method likely diverging.")
            return

    print("\nMax iterations reached.")

def modified_secant(func_str, x0, delta, tol, max_iter):
    print_header("Modified Secant")
    print(f"Using perturbation delta = {delta}")
    print(f"{'Iter':<10} | {'Root Est (x1)':<18} | {'f(x1)':<18} | {'Error':<18}")
    print("-" * 70)

    for i in range(1, max_iter + 1):
        f0 = safe_eval(func_str, x0)
        f0_delta = safe_eval(func_str, x0 + delta * x0)
        
        denom = f0_delta - f0
        if denom == 0:
            print("Error: Division by zero in modified secant.")
            return

        # Formula: x_new = x - (delta * x * f(x)) / (f(x + delta*x) - f(x))
        x1 = x0 - (delta * x0 * f0) / denom
        f1 = safe_eval(func_str, x1)
        error = abs(x1 - x0)
        
        print_row(i, x1, f1, error)
        
        if abs(f1) < tol or error < tol:
            print(f"\nConverged to root: {x1:.8f} after {i} iterations.")
            return
        
        x0 = x1

    print("\nMax iterations reached.")

# --- MAIN MENU ---

def main():
    while True:
        print("\n" + "="*40)
        print(" ZERO OF FUNCTIONS (ZOF) SOLVER - CLI")
        print("="*40)
        print("1. Bisection Method")
        print("2. Regula Falsi Method")
        print("3. Secant Method")
        print("4. Newton-Raphson Method")
        print("5. Fixed Point Iteration")
        print("6. Modified Secant Method")
        print("7. Exit")
        
        choice = input("\nSelect Method (1-7): ")
        
        if choice == '7':
            print("Exiting...")
            break
        
        if choice not in ['1','2','3','4','5','6']:
            print("Invalid choice.")
            continue

        # Common Inputs
        if choice == '5':
            print("Note: For Fixed Point, enter g(x) such that x = g(x).")
            func_input = input("Enter g(x): ")
        else:
            func_input = input("Enter function f(x) (e.g., x**2 - 4): ")
            
        tol = float(input("Enter tolerance (e.g., 0.0001): "))
        max_iter = int(input("Enter max iterations: "))

        if choice == '1': # Bisection
            a = float(input("Enter initial guess a (lower bound): "))
            b = float(input("Enter initial guess b (upper bound): "))
            bisection(func_input, a, b, tol, max_iter)
            
        elif choice == '2': # Regula Falsi
            a = float(input("Enter initial guess a (lower bound): "))
            b = float(input("Enter initial guess b (upper bound): "))
            regula_falsi(func_input, a, b, tol, max_iter)
            
        elif choice == '3': # Secant
            x0 = float(input("Enter first guess x0: "))
            x1 = float(input("Enter second guess x1: "))
            secant(func_input, x0, x1, tol, max_iter)
            
        elif choice == '4': # Newton
            deriv_input = input("Enter derivative f'(x) (e.g., 2*x): ")
            x0 = float(input("Enter initial guess x0: "))
            newton_raphson(func_input, deriv_input, x0, tol, max_iter)
            
        elif choice == '5': # Fixed Point
            x0 = float(input("Enter initial guess x0: "))
            fixed_point(func_input, x0, tol, max_iter)

        elif choice == '6': # Mod Secant
            x0 = float(input("Enter initial guess x0: "))
            delta = float(input("Enter perturbation delta (e.g., 0.01): "))
            modified_secant(func_input, x0, delta, tol, max_iter)
            
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()