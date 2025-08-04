# interface/rpy2_init.py
"""
RPy2 initialization module - handles R interface setup with proper context management
Fixed version for Streamlit compatibility
"""

import warnings
import contextvars
import threading
import sys
import os

# Suppress the R main thread warning for Streamlit
warnings.filterwarnings("ignore", message="R is not initialized by the main thread")

# Global variables to store initialized objects
_libstable4u = None
_alphastable = None
_qcv_test = None
_initialized = False
_lock = threading.Lock()
_conversion_context = contextvars.ContextVar('rpy2_conversions', default=None)

def get_rpy2_version():
    """Check RPy2 version for compatibility"""
    try:
        import rpy2
        version = rpy2.__version__
        major, minor = map(int, version.split('.')[:2])
        return major, minor, version
    except Exception as e:
        print(f"Could not determine RPy2 version: {e}")
        return None, None, None

def ensure_rpy2_context():
    """Ensure RPy2 conversions are active in current thread context"""
    try:
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2 import robjects
        
        major, minor, version = get_rpy2_version()
        print(f"RPy2 version: {version}")
        
        # Handle different RPy2 versions
        if major is None:
            return False
            
        # For RPy2 3.x, use different activation methods
        if major >= 3:
            # Try new activation method first
            try:
                if hasattr(pandas2ri, 'activate'):
                    pandas2ri.activate()
                if hasattr(numpy2ri, 'activate'):
                    numpy2ri.activate()
                print("✅ RPy2 conversions activated (new method)")
                return True
            except Exception as e:
                print(f"New activation method failed: {e}")
                
            # Try alternative activation
            try:
                if hasattr(robjects.conversion, 'localconverter'):
                    # Store in context variable
                    ctx = robjects.conversion.localconverter(
                        robjects.default_converter + pandas2ri.converter + numpy2ri.converter
                    )
                    _conversion_context.set(ctx)
                    print("✅ RPy2 conversions set via localconverter")
                    return True
            except Exception as e:
                print(f"Localconverter method failed: {e}")
        
        # Fallback for older versions
        try:
            if hasattr(pandas2ri, 'ri2py'):
                robjects.pandas2ri.activate()
            if hasattr(numpy2ri, 'ri2py'):
                robjects.numpy2ri.activate()
            print("✅ RPy2 conversions activated (legacy method)")
            return True
        except Exception as e:
            print(f"Legacy activation failed: {e}")
            
        return False
        
    except ImportError as e:
        print(f"RPy2 import error: {e}")
        return False
    except Exception as e:
        print(f"Error setting up RPy2 conversions: {e}")
        return False

def create_qcv_function():
    """Create QCV function with proper context"""
    try:
        from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
        from rpy2 import robjects
        
        # Ensure conversions are active before creating R function
        if not ensure_rpy2_context():
            return None
        
        qcv_r_code = """
        qcv_stat <- function(x) {
          tryCatch({
            x <- sort((x - mean(x)) / sd(x))
            q25 <- quantile(x, 0.25)
            q75 <- quantile(x, 0.75)
            var_left <- var(x[x < q25])
            var_right <- var(x[x > q75])
            var_mid <- var(x[x > q25 & x < q75])
            qcv = (var_left + var_right) / (2 * var_mid)
            return(qcv)
          }, error = function(e) {
            return(1.0)
          })
        }
        """
        
        # Use conversion context if available
        ctx = _conversion_context.get()
        if ctx:
            with ctx:
                return SignatureTranslatedAnonymousPackage(qcv_r_code, "qcv_test")
        else:
            return SignatureTranslatedAnonymousPackage(qcv_r_code, "qcv_test")
        
    except Exception as e:
        print(f"Error creating QCV function: {e}")
        return None

def initialize_rpy2():
    """Initialize RPy2 conversions and R packages with proper context handling"""
    global _libstable4u, _alphastable, _qcv_test, _initialized
    
    with _lock:
        if _initialized:
            # Ensure context is active even if already initialized
            ensure_rpy2_context()
            return _libstable4u, _alphastable, _qcv_test
        
        try:
            print("🔄 Initializing RPy2...")
            
            # Ensure conversions are active
            if not ensure_rpy2_context():
                print("⚠️ Could not set up RPy2 conversions, continuing without R interface")
                _initialized = True
                return None, None, None
            
            # Import R packages with better error handling
            try:
                from alpha_stable_mixture.r_interface import libstable4u, alphastable
                print("✅ R packages imported successfully")
            except ImportError as e:
                print(f"Warning: Could not import R packages: {e}")
                _initialized = True
                return None, None, None
            except Exception as e:
                print(f"Warning: Error importing R packages: {e}")
                _initialized = True
                return None, None, None
            
            # Create QCV test function with proper context
            print("🔄 Creating QCV test function...")
            qcv_test = create_qcv_function()
            if qcv_test is not None:
                print("✅ QCV test function created successfully")
            else:
                print("⚠️ QCV test function creation failed, will use Python fallback")
            
            # Store globally
            _libstable4u = libstable4u
            _alphastable = alphastable
            _qcv_test = qcv_test
            _initialized = True
            
            print("✅ RPy2 initialization completed successfully")
            return libstable4u, alphastable, qcv_test
            
        except Exception as e:
            print(f"Warning: Could not initialize R interface: {e}")
            _initialized = True
            return None, None, None

def get_r_objects_with_context():
    """Get R objects ensuring proper context is set"""
    # Ensure conversions are active in current thread
    ensure_rpy2_context()
    
    # Return initialized objects
    if _initialized:
        return _libstable4u, _alphastable, _qcv_test, get_float_vector()
    else:
        libstable4u, alphastable, qcv_test = initialize_rpy2()
        return libstable4u, alphastable, qcv_test, get_float_vector()

def get_float_vector():
    """Get FloatVector with proper context"""
    try:
        ensure_rpy2_context()
        from rpy2.robjects import FloatVector
        return FloatVector
    except Exception as e:
        print(f"Error getting FloatVector: {e}")
        return None

def run_with_r_context(func, *args, **kwargs):
    """Run a function with proper R context - improved version"""
    try:
        # Check if we have a conversion context
        ctx = _conversion_context.get()
        
        if ctx:
            # Use the stored conversion context
            with ctx:
                return func(*args, **kwargs)
        else:
            # Ensure conversions are active and run directly
            if ensure_rpy2_context():
                return func(*args, **kwargs)
            else:
                raise RuntimeError("Could not establish R context")
                
    except Exception as e:
        print(f"Error running function with R context: {e}")
        # Try one more time with fresh context setup
        try:
            print("🔄 Attempting to re-establish R context...")
            if ensure_rpy2_context():
                return func(*args, **kwargs)
            else:
                raise RuntimeError("Failed to re-establish R context")
        except Exception as e2:
            print(f"Final attempt failed: {e2}")
            raise e2

def check_r_availability():
    """Check if R interface is properly available"""
    try:
        libstable4u, alphastable, qcv_test, FloatVector = get_r_objects_with_context()
        
        if libstable4u is None or alphastable is None or FloatVector is None:
            return False, "R packages not available"
            
        # Test a simple R operation
        test_data = FloatVector([1, 2, 3, 4, 5])
        if test_data is None:
            return False, "FloatVector creation failed"
            
        return True, "R interface working properly"
        
    except Exception as e:
        return False, f"R interface error: {e}"

# Initialize once when module is imported (but don't fail if it doesn't work)
try:
    print("🚀 Starting RPy2 initialization...")
    libstable4u, alphastable, qcv_test = initialize_rpy2()
    
    # Check if initialization was successful
    is_available, status = check_r_availability()
    if is_available:
        print("✅ RPy2 initialization completed successfully")
    else:
        print(f"⚠️ RPy2 initialization had issues: {status}")
        
except Exception as e:
    print(f"⚠️ RPy2 initialization failed: {e}")
    libstable4u, alphastable, qcv_test = None, None, None