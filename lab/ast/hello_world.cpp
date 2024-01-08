#include<iostream>
using namespace std;

int main(){
    cout<<"hello world!"<<endl;
    return 0;
}

/*
  FUNCTION_DECL (main)
    COMPOUND_STMT ()
      CALL_EXPR (operator<<)
        CALL_EXPR (operator<<)
          DECL_REF_EXPR (cout)
          UNEXPOSED_EXPR (operator<<)
            DECL_REF_EXPR (operator<<)
          UNEXPOSED_EXPR ()
            STRING_LITERAL ("hello world!")
        UNEXPOSED_EXPR (operator<<)
          DECL_REF_EXPR (operator<<)
        UNEXPOSED_EXPR (endl)
          DECL_REF_EXPR (endl)
      RETURN_STMT ()
        INTEGER_LITERAL ()
*/