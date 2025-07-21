// Problem: Given an integer array nums , return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i] . 
// The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer. 
// You must write an algorithm that runs in O(n) time .
// Example: 
// ``` 
// Input: nums = [1,2,3,4] 
// Output: [24,12,8,6] 


#include <bits/stdc++.h> 
using namespace std;

vector <int> sol(vector <int> nums)
{
    int n = nums.size();
    vector <int> answer(n, 1);

    int mul = 1;
    for (int i = 0; i < nums.size(); i++){
        mul *= nums[i];
    }
    
    for (int i = 0; i < nums.size(); i++){
        answer[i] = mul / nums[i];
    }

    return answer;
}

int main(){


    vector <int> v = {1, 2, 3, 4}, res;
    res = sol(v);

    for (int i = 0; i < res.size(); i++){
        cout<<res[i]<<" ";
    }

    return 0;
}