# C++ Header and Source Ordering

Use the following order for new code and when reorganizing existing code.

## Files

1. File documentation comment, when the file has one.
2. `#pragma once` in headers.
3. Includes: in headers, standard-library headers first and then project
   headers. In source files, include the corresponding header first, then
   project headers and standard-library headers. Keep groups together and avoid
   relying on transitive includes.
4. Declarations or definitions, grouped by the public API's logical sections.

## Class declarations

Declare private data members first, in the order they are initialized. Follow
them with the public interface in this order:

1. Constructors: default, ordinary/converting, copy, then move.
2. Destructor.
3. Assignment operators: copy, then move.
4. Observers and accessors, including iterators.
5. Modifiers.
6. Element-access and comparison operators.
7. Any remaining public operations, grouped by responsibility.

## Source definitions

- Define class members in the same order as their declarations.
- Keep free-function definitions in the order their declarations appear in the
  corresponding public header.
- In a header, declare template functions before non-template functions.
- In CUDA implementation files, keep internal kernels/helpers before the
  public host-side wrappers; keep wrappers in public-header order.
- Put explicit template instantiations at the end of the source file.

## Include dependencies

A header includes the standard-library declarations it uses directly; do not
depend on another header to include them transitively.
