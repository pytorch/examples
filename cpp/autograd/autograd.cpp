#include <torch/torch.h>
#include <iostream>

using namespace torch::autograd;

void basic_autograd_operations_example() {
  std::cout << "====== Running: \"Basic autograd operations\" ======" << std::endl;

  // Create a tensor and set ``torch::requires_grad()`` to track computation with it
  auto x = torch::ones({2, 2}, torch::requires_grad());
  std::cout << x << std::endl;

  // Do a tensor operation:
  auto y = x + 2;
  std::cout << y << std::endl;

  // ``y`` was created as a result of an operation, so it has a ``grad_fn``.
  std::cout << y.grad_fn()->name() << std::endl;

  // Do more operations on ``y``
  auto z = y * y * 3;
  auto out = z.mean();

  std::cout << z << std::endl;
  std::cout << z.grad_fn()->name() << std::endl;
  std::cout << out << std::endl;
  std::cout << out.grad_fn()->name() << std::endl;

  // ``.requires_grad_( ... )`` changes an existing tensor's ``requires_grad`` flag in-place.
  auto a = torch::randn({2, 2});
  a = ((a * 3) / (a - 1));
  std::cout << a.requires_grad() << std::endl;

  a.requires_grad_(true);
  std::cout << a.requires_grad() << std::endl;

  auto b = (a * a).sum();
  std::cout << b.grad_fn()->name() << std::endl;

  // Let's backprop now. Because ``out`` contains a single scalar, ``out.backward()``
  // is equivalent to ``out.backward(torch::tensor(1.))``.
  out.backward();

  // Print gradients d(out)/dx
  std::cout << x.grad() << std::endl;

  // Now let's take a look at an example of vector-Jacobian product:
  x = torch::randn(3, torch::requires_grad());

  y = x * 2;
  while (y.norm().item<double>() < 1000) {
    y = y * 2;
  }

  std::cout << y << std::endl;
  std::cout << y.grad_fn()->name() << std::endl;

  // If we want the vector-Jacobian product, pass the vector to ``backward`` as argument:
  auto v = torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat);
  y.backward(v);

  std::cout << x.grad() << std::endl;

  // You can also stop autograd from tracking history on tensors that require gradients
  // either by putting ``torch::NoGradGuard`` in a code block
  std::cout << x.requires_grad() << std::endl;
  std::cout << x.pow(2).requires_grad() << std::endl;

  {
    torch::NoGradGuard no_grad;
    std::cout << x.pow(2).requires_grad() << std::endl;
  }

  // Or by using ``.detach()`` to get a new tensor with the same content but that does
  // not require gradients:
  std::cout << x.requires_grad() << std::endl;
  y = x.detach();
  std::cout << y.requires_grad() << std::endl;
  std::cout << x.eq(y).all().item<bool>() << std::endl;
}

void compute_higher_order_gradients_example() {
  std::cout << "====== Running \"Computing higher-order gradients in C++\" ======" << std::endl;

  // One of the applications of higher-order gradients is calculating gradient penalty.
  // Let's see an example of it using ``torch::autograd::grad``:

  auto model = torch::nn::Linear(4, 3);

  auto input = torch::randn({3, 4}).requires_grad_(true);
  auto output = model(input);

  // Calculate loss
  auto target = torch::randn({3, 3});
  auto loss = torch::nn::MSELoss()(output, target);

  // Use norm of gradients as penalty
  auto grad_output = torch::ones_like(output);
  auto gradient = torch::autograd::grad({output}, {input}, /*grad_outputs=*/{grad_output}, /*create_graph=*/true)[0];
  auto gradient_penalty = torch::pow((gradient.norm(2, /*dim=*/1) - 1), 2).mean();

  // Add gradient penalty to loss
  auto combined_loss = loss + gradient_penalty;
  combined_loss.backward();

  std::cout << input.grad() << std::endl;
}

// Inherit from Function
class LinearFunction : public Function<LinearFunction> {
 public:
  // Note that both forward and backward are static functions

  // bias is an optional argument
  static torch::Tensor forward(
      AutogradContext *ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor bias = torch::Tensor()) {
    ctx->save_for_backward({input, weight, bias});
    auto output = input.mm(weight.t());
    if (bias.defined()) {
      output += bias.unsqueeze(0).expand_as(output);
    }
    return output;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto weight = saved[1];
    auto bias = saved[2];

    auto grad_output = grad_outputs[0];
    auto grad_input = grad_output.mm(weight);
    auto grad_weight = grad_output.t().mm(input);
    auto grad_bias = torch::Tensor();
    if (bias.defined()) {
      grad_bias = grad_output.sum(0);
    }

    return {grad_input, grad_weight, grad_bias};
  }
};

class MulConstant : public Function<MulConstant> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor tensor, double constant) {
    // ctx is a context object that can be used to stash information
    // for backward computation
    ctx->saved_data["constant"] = constant;
    return tensor * constant;
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    // We return as many input gradients as there were arguments.
    // Gradients of non-tensor arguments to forward must be `torch::Tensor()`.
    return {grad_outputs[0] * ctx->saved_data["constant"].toDouble(), torch::Tensor()};
  }
};

void custom_autograd_function_example() {
  std::cout << "====== Running \"Using custom autograd function in C++\" ======" << std::endl;
  {
    auto x = torch::randn({2, 3}).requires_grad_();
    auto weight = torch::randn({4, 3}).requires_grad_();
    auto y = LinearFunction::apply(x, weight);
    y.sum().backward();

    std::cout << x.grad() << std::endl;
    std::cout << weight.grad() << std::endl;
  }
  {
    auto x = torch::randn({2}).requires_grad_();
    auto y = MulConstant::apply(x, 5.5);
    y.sum().backward();

    std::cout << x.grad() << std::endl;
  }
}

int main() {
  std::cout << std::boolalpha;

  basic_autograd_operations_example();

  std::cout << "\n";

  compute_higher_order_gradients_example();

  std::cout << "\n";

  custom_autograd_function_example();
}
