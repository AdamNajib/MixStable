# interface/rpy2_init.py
"""
RPy2 initialization module - handles R interface setup with proper context management
Fixed version for Streamlit compatibility with threading context issues
"""

import warnings
import contextvars
import threading
import sys
import os
import copy

# Suppress the R main thread warning for Streamlit
warnings.filterwarnings("ignore", message="R is not initialized by the main thread")

# Global variables to store initialized objects
_libstable4u = None
_alphastable = None
_qcv_test = None
_initialized = False
_lock = threading.Lock()
_main_converter = None

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

def setup_global_converter():
    """Set up a global converter that persists across threads"""
    global _main_converter
    
    try:
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2 import robjects
        
        print("🔧 Setting up global RPy2 converter...")
        
        # Create a combined converter
        converter = robjects.default_converter
        
        # Add pandas and numpy converters
        try:
            converter = converter + pandas2ri.converter
            print("✅ Added pandas2ri converter")
        except Exception as e:
            print(f"⚠️ Could not add pandas2ri converter: {e}")
        
        try:
            converter = converter + numpy2ri.converter
            print("✅ Added numpy2ri converter")
        except Exception as e:
            print(f"⚠️ Could not add numpy2ri converter: {e}")
        
        # Store globally
        _main_converter = converter
        
        # Set as default converter
        robjects.conversion.set_conversion(converter)
        print("✅ Global converter set successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to set up global converter: {e}")
        return False

def ensure_rpy2_context():
    """Ensure RPy2 conversions are active in current thread context"""
    global _main_converter
    
    try:
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2 import robjects
        
        # If we have a global converter, use it
        if _main_converter is not None:
            try:
                robjects.conversion.set_conversion(_main_converter)
                print(f"✅ Thread {threading.current_thread().name}: Using global converter")
                return True
            except Exception as e:
                print(f"⚠️ Could not set global converter in thread: {e}")
        
        # Fallback: try to recreate converter in this thread
        major, minor, version = get_rpy2_version()
        if major is None:
            return False
            
        print(f"RPy2 version: {version}")
        
        # For RPy2 3.x, use activation methods
        if major >= 3:
            try:
                # Method 1: Direct activation
                if hasattr(pandas2ri, 'activate'):
                    pandas2ri.activate()
                if hasattr(numpy2ri, 'activate'):
                    numpy2ri.activate()
                print(f"✅ Thread {threading.current_thread().name}: Direct activation succeeded")
                return True
            except Exception as e:
                print(f"Direct activation failed: {e}")
            
            try:
                # Method 2: Manual converter setup
                converter = robjects.default_converter + pandas2ri.converter + numpy2ri.converter
                robjects.conversion.set_conversion(converter)
                print(f"✅ Thread {threading.current_thread().name}: Manual converter setup succeeded")
                return True
            except Exception as e:
                print(f"Manual converter setup failed: {e}")
        
        # Legacy fallback
        try:
            robjects.pandas2ri.activate()
            robjects.numpy2ri.activate()
            print(f"✅ Thread {threading.current_thread().name}: Legacy activation succeeded")
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
        
        return SignatureTranslatedAnonymousPackage(qcv_r_code, "qcv_test")
        
    except Exception as e:
        print(f"Error creating QCV function: {e}")
        return None

def initialize_rpy2():
    """Initialize RPy2 conversions and R packages with proper context handling"""
    global _libstable4u, _alphastable, _qcv_test, _initialized
    
    with _lock:
        if _initialized:
            # Re-ensure context is active for this thread
            ensure_rpy2_context()
            return _libstable4u, _alphastable, _qcv_test
        
        try:
            print("🔄 Initializing RPy2...")
            
            # Set up the global converter first
            if not setup_global_converter():
                print("⚠️ Could not set up global converter, continuing with thread-local setup")
            
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
    """Run a function with proper R context - improved version for threading"""
    try:
        # Always ensure context is set up for current thread
        print(f"🔧 Thread {threading.current_thread().name}: Setting up R context...")
        
        if not ensure_rpy2_context():
            raise RuntimeError("Could not establish R context")
        
        print(f"🔧 Thread {threading.current_thread().name}: R context established, running function...")
        return func(*args, **kwargs)
                
    except Exception as e:
        print(f"❌ Thread {threading.current_thread().name}: Error running function with R context: {e}")
        
        # Try one more time with fresh context setup
        try:
            print("🔄 Attempting to re-establish R context...")
            
            # Force re-setup of converter
            if setup_global_converter() and ensure_rpy2_context():
                print("🔧 Re-established R context, trying function again...")
                return func(*args, **kwargs)
            else:
                raise RuntimeError("Failed to re-establish R context")
                
        except Exception as e2:
            print(f"❌ Final attempt failed: {e2}")
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

def force_reinitialize():
    """Force re-initialization of RPy2 (useful for threading issues)"""
    global _initialized, _main_converter
    print("🔄 Forcing RPy2 re-initialization...")
    
    with _lock:
        _initialized = False
        _main_converter = None
        
        # Re-initialize
        return initialize_rpy2()

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