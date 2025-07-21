// Task: Implement a class Stack with the following methods:
// push(item)
// pop()
// peek()
// is_empty()



#include <bits/stdc++.h> 
#include <iterator>
using namespace std;

class myStack{
    vector <int> mynums;

public:
    void push(int elem)
    {
        mynums.push_back(elem);
    }

    vector <int> pop()
    {
        mynums.pop_back();
    }

    int peek(){
        int n = mynums.size();
        if (is_empty() == false){
            return mynums[n-1];
        }
         
        return 1e9; // a big int as the stack is empty
    }

    bool is_empty()
    {
        int n = mynums.size();

        if (n == 0){
            return true;
        }

        return false;
    }

};


int main(){

    myStack st;
    cout<<st.is_empty();
    
    return 0;
}