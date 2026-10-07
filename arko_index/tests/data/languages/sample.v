// A counter with synchronous reset.
`timescale 1ns / 1ps

module counter #(
    parameter WIDTH = 8
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             enable,
    output reg  [WIDTH-1:0] count
);

  // Counts up while enabled.
  always @(posedge clk) begin
    if (reset)
      count <= {WIDTH{1'b0}};
    else if (enable)
      count <= count + 1'b1;
  end

endmodule

module top (
    input wire clk,
    input wire rst,
    output wire [7:0] leds
);

  wire tick;

  counter #(.WIDTH(8)) u_counter (
      .clk(clk),
      .reset(rst),
      .enable(tick),
      .count(leds)
  );

  function automatic [7:0] double(input [7:0] x);
    double = x << 1;
  endfunction

  task show;
    $display("leds = %b", leds);
  endtask

  assign tick = 1'b1;

endmodule
