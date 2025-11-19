import math
from flask import Flask, render_template, request

app = Flask(__name__)

# --- 1. HELPER FUNCTIONS ---

def safe_eval(func_str, x):
    """Evaluates math string safely."""
    allowed_locals = math.__dict__
    allowed_locals['x'] = x
    try:
        # Replace ^ with ** for user friendliness if they use caret for power
        func_str = func_str.replace('^', '**')
        return eval(func_str, {"__builtins__": {}}, allowed_locals)
    except Exception:
        return None

# --- 2. SOLVER ALGORITHMS (Returning Data Lists) ---

def solve_bisection(func_str, a, b, tol, max_iter):
    data = []
    fa = safe_eval(func_str, a)
    fb = safe_eval(func_str, b)

    if fa is None or fb is None:
        return {"error": "Invalid function or value."}
    if fa * fb >= 0:
        return {"error": "Bisection fails: f(a) and f(b) must have opposite signs."}

    root = None
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = safe_eval(func_str, c)
        error = abs(b - a)
        
        data.append({
            "iter": i, 
            "root": round(c, 8), 
            "func": round(fc, 8), 
            "error": round(error, 8)
        })

        if abs(fc) < tol or error < tol:
            root = c
            break

        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc
            
    return {"data": data, "root": root, "converged": root is not None}

def solve_regula_falsi(func_str, a, b, tol, max_iter):
    data = []
    fa = safe_eval(func_str, a)
    fb = safe_eval(func_str, b)

    if fa * fb >= 0:
        return {"error": "Regula Falsi fails: f(a) and f(b) must have opposite signs."}

    root = None
    prev_c = a
    for i in range(1, max_iter + 1):
        c = (a * fb - b * fa) / (fb - fa)
        fc = safe_eval(func_str, c)
        error = abs(c - prev_c)
        
        data.append({
            "iter": i, 
            "root": round(c, 8), 
            "func": round(fc, 8), 
            "error": round(error, 8)
        })

        if abs(fc) < tol or error < tol:
            root = c
            break

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        prev_c = c

    return {"data": data, "root": root, "converged": root is not None}

def solve_secant(func_str, x0, x1, tol, max_iter):
    data = []
    root = None
    
    for i in range(1, max_iter + 1):
        f0 = safe_eval(func_str, x0)
        f1 = safe_eval(func_str, x1)
        
        if f0 is None or f1 is None: return {"error": "Eval error"}
        if (f1 - f0) == 0: return {"error": "Division by zero (f1 - f0 = 0)"}

        x2 = x1 - (f1 * (x1 - x0)) / (f1 - f0)
        f2 = safe_eval(func_str, x2)
        error = abs(x2 - x1)

        data.append({
            "iter": i, 
            "root": round(x2, 8), 
            "func": round(f2, 8), 
            "error": round(error, 8)
        })

        if abs(f2) < tol or error < tol:
            root = x2
            break
        
        x0, x1 = x1, x2

    return {"data": data, "root": root, "converged": root is not None}

def solve_newton(func_str, deriv_str, x0, tol, max_iter):
    data = []
    root = None

    for i in range(1, max_iter + 1):
        f0 = safe_eval(func_str, x0)
        f_prime0 = safe_eval(deriv_str, x0)

        if f_prime0 == 0: return {"error": "Derivative is zero. Method fails."}
        if f0 is None or f_prime0 is None: return {"error": "Eval error"}

        x1 = x0 - (f0 / f_prime0)
        f1 = safe_eval(func_str, x1)
        error = abs(x1 - x0)

        data.append({
            "iter": i, 
            "root": round(x1, 8), 
            "func": round(f1, 8), 
            "error": round(error, 8)
        })

        if abs(f1) < tol or error < tol:
            root = x1
            break
        x0 = x1

    return {"data": data, "root": root, "converged": root is not None}

def solve_fixed_point(g_str, x0, tol, max_iter):
    data = []
    root = None

    for i in range(1, max_iter + 1):
        x1 = safe_eval(g_str, x0)
        if x1 is None: return {"error": "Eval error"}
        
        error = abs(x1 - x0)
        
        data.append({
            "iter": i, 
            "root": round(x1, 8), 
            "func": round(x1, 8), # g(x) is the new x
            "error": round(error, 8)
        })

        if error < tol:
            root = x1
            break
        
        x0 = x1
        if x0 > 1e10: return {"error": "Divergence detected."}

    return {"data": data, "root": root, "converged": root is not None}

def solve_mod_secant(func_str, x0, delta, tol, max_iter):
    data = []
    root = None

    for i in range(1, max_iter + 1):
        f0 = safe_eval(func_str, x0)
        f0_delta = safe_eval(func_str, x0 + delta * x0)
        
        denom = f0_delta - f0
        if denom == 0: return {"error": "Division by zero"}

        x1 = x0 - (delta * x0 * f0) / denom
        f1 = safe_eval(func_str, x1)
        error = abs(x1 - x0)

        data.append({
            "iter": i, 
            "root": round(x1, 8), 
            "func": round(f1, 8), 
            "error": round(error, 8)
        })

        if abs(f1) < tol or error < tol:
            root = x1
            break
        x0 = x1

    return {"data": data, "root": root, "converged": root is not None}

# --- 3. ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    
    if request.method == 'POST':
        try:
            # Get common inputs
            method = request.form.get('method')
            func_str = request.form.get('function')
            tol = float(request.form.get('tolerance'))
            max_iter = int(request.form.get('max_iter'))
            
            # Dispatch based on method
            if method == 'bisection':
                a = float(request.form.get('param_a'))
                b = float(request.form.get('param_b'))
                result = solve_bisection(func_str, a, b, tol, max_iter)
                
            elif method == 'regula_falsi':
                a = float(request.form.get('param_a'))
                b = float(request.form.get('param_b'))
                result = solve_regula_falsi(func_str, a, b, tol, max_iter)
                
            elif method == 'secant':
                x0 = float(request.form.get('param_x0'))
                x1 = float(request.form.get('param_x1'))
                result = solve_secant(func_str, x0, x1, tol, max_iter)
                
            elif method == 'newton':
                deriv_str = request.form.get('derivative')
                x0 = float(request.form.get('param_x0'))
                result = solve_newton(func_str, deriv_str, x0, tol, max_iter)
                
            elif method == 'fixed_point':
                x0 = float(request.form.get('param_x0'))
                result = solve_fixed_point(func_str, x0, tol, max_iter)
                
            elif method == 'mod_secant':
                x0 = float(request.form.get('param_x0'))
                delta = float(request.form.get('param_delta'))
                result = solve_mod_secant(func_str, x0, delta, tol, max_iter)
                
        except ValueError:
            result = {"error": "Input Error: Ensure all fields are numbers where required."}
        except Exception as e:
            result = {"error": f"System Error: {str(e)}"}

    return render_template('index.html', result=result)

if __name__ == '__main__':
    # debug=True is fine for local dev, but we'll handle deployment later
    app.run(debug=True)