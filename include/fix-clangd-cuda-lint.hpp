// fix clangd lint about cuda
#ifdef __clang__
#undef __noinline__
#endif
