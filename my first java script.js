
function sub(a, b) {
    let sum=a+b
    let product=a*b

    return {sum, product}
}

// arrow functions

const divide=(a,b)=>{
    if(b==0){
        return {message:"Please you can't divide by zero", div:null,error:true}
    }
    let div=a/b;
    return {message:null,div,error:false};

}

const {message, div,error}=divide(2,0);
if(error){
    console.log(message);
}else{
    console.log(`The quotient is : ${div}`);
}



// let {sum, product} =sub(5,3)
// console.log("the difference between 5 and 3 is ", sum, "and the product is: ", product)