#ifndef _SHADER_PARAMS_H
#define _SHADER_PARAMS_H

// This file contains the struct types that are used for passing parameters to
// a metal shader. Qny parameters other than the data buffers will
// be packaged inside a struct.
//
// Since both the C++ file containing the code that calls the shader and
// corresponding shader code in the .metal file need to refer to the same struct,
// Let's avoid code duplication by defining them in one place (i.e., this file)!

// We could optionally create a "shader_types" namespace here to contain our structs,
// but maybe it's not needed for such a small project.

// Used by the matrix multiplication shaders.
typedef struct
{
    unsigned int row_dim_x; // Number of rows in X
    unsigned int col_dim_x; // Number of columns in X
    unsigned int inner_dim; // Number of columsn in A = number of rows in B
} MatMulParams;

#endif /* _SHADER_PARAMS_H */
