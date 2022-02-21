`timescale 1ns/10ps

module  CONV(
    input		clk,
    input		reset,
    output reg 	busy,
    input		ready,

    output [11:0]	iaddr,
    input [19:0]	idata,

    output	cwr,
    output reg [11:0]	caddr_wr,
    output reg [19:0]	cdata_wr,

    output	crd,
    output reg [11:0]	caddr_rd,
    input [19:0]	cdata_rd,

    output reg [2:0]	csel
  );

  //state
parameter IDLE=4'd0,
          READ_9=4'd1,
          READ_C=4'd2,
          CONV=4'd3,
          RELU=4'd4,
          LAYER0_WRITE=4'd5,
          LAYER0_DONE=4'd6,
          MAX_POOLING=4'd7,
          LAYER1_WRITE=4'd8,
          LAYER1_DONE=4'd9; 

reg [3:0] state,n_state;
reg [6:0] row,col;
reg [6:0] next_row,next_col;
reg [6:0] tmp_row,tmp_col;
reg [3:0] read_counter,next_read_counter;
reg [3:0] conv_count,next_conv_count;
reg [19:0] reg0,reg1,reg2,reg3,reg4,reg5,reg6,reg7,reg8;
wire read_en,conv_en;
wire read9,readc,relu;
wire max_en,layer1_wen;
wire zero_padding;
wire [19:0] tmp_data;
wire [6:0] img_row,img_col;
reg [19:0] weight,img_data;
reg signed [39:0] psum,next_psum,addBias_result;
wire signed [39:0] tmp_result;
reg [19:0] conv_data,next_conv_data;
reg [11:0] caddr_count,next_caddr_count;
reg [19:0] max,n_max;
wire [11:0] max_addr;

//control signal
assign read9=(state==READ_9)?1'b1:1'b0;
assign readc=(state==READ_C)?1'b1:1'b0;
assign relu=(state==RELU)?1'b1:1'b0;
assign max_en=(state==MAX_POOLING)?1'b1:1'b0;
assign layer1_wen=(state==LAYER1_WRITE)?1'b1:1'b0;

assign read_en=(read9|readc|max_en); //control read times
assign conv_en=(state==CONV)?1'b1:1'b0; //control conv times
assign zero_padding=(tmp_row==7'd0||tmp_row==7'd65||tmp_col==7'd0||tmp_col==7'd65)?1'b1:1'b0;
assign cwr=(state==LAYER0_WRITE||state==LAYER1_WRITE)?1'b1:1'b0;
assign crd=(state==MAX_POOLING)?1'b1:1'b0;

assign img_row=tmp_row-7'd1;
assign img_col=tmp_col-7'd1;
assign tmp_data=(zero_padding)?20'd0:idata;
assign iaddr=(zero_padding)?12'd0:(img_row<<6)+img_col;
assign tmp_result=$signed(img_data)*$signed(weight);
assign max_addr=(row<<6)+col;

//FSM
always@(posedge clk or posedge reset)begin
  if(reset)begin
    state<=IDLE;
  end
  else begin
    state<=n_state;
  end
end
always@(*)begin
  case(state)
      IDLE:begin
          n_state=READ_9;
      end
      READ_9:begin
        if(read_counter==4'd8)begin
          n_state=CONV;
        end
        else begin
          n_state=state;
        end
      end
      READ_C:begin
        if(read_counter==4'd2)begin
          n_state=CONV;
        end
        else begin
          n_state=state;
        end
      end
      CONV:begin
        if(conv_count==4'd8)begin
          n_state=RELU;
        end
        else begin
          n_state=state;
        end
      end
      RELU:begin
        n_state=LAYER0_WRITE;
      end
      LAYER0_WRITE:begin
        if(caddr_wr==12'd4095)begin
          n_state=LAYER0_DONE;
        end
        else if(col==7'd63)begin
          n_state=READ_9;
        end
        else begin
          n_state=READ_C;
        end
      end
      LAYER0_DONE:begin
        n_state=MAX_POOLING;
      end
      MAX_POOLING:begin
        if(read_counter==4'd3)begin
          n_state=LAYER1_WRITE;
        end
        else begin
          n_state=state;
        end
      end
      LAYER1_WRITE:begin
        if(caddr_wr==12'd1023)begin
          n_state=LAYER1_DONE;
        end
        else begin
          n_state=MAX_POOLING;
        end
      end
      LAYER1_DONE:begin
        n_state=LAYER1_DONE;
      end
      default:begin
        n_state=IDLE;
      end
  endcase
end

//read_counter
always@(posedge clk or posedge reset)begin
  if(reset)begin
    read_counter<=4'd0;
  end
  else if(read_en)begin
    read_counter<=next_read_counter;
  end
  else begin
    read_counter<=4'd0;
  end
end

always@(*)begin
  if(read_en)begin
    next_read_counter=read_counter+4'd1;
  end
  else begin
    next_read_counter=read_counter;
  end
end

// define row and column
always@(posedge clk or posedge reset)begin
  if(reset)begin
    row<=7'd0;
    col<=7'd0;
  end
  else begin
    row<=next_row;
    col<=next_col;
  end
end

always@(*)begin
  if(state==LAYER0_WRITE)begin
    if(row==7'd63 && col==7'd63)begin
      next_row=7'd0;
      next_col=7'd0;
    end
    else if(col==7'd63)begin
      next_row=row+7'd1;
      next_col=7'd0;
    end
    else begin
      next_row=row;
      next_col=col+7'd1;
    end
  end
  else if(state==LAYER1_WRITE)begin
    if(col==7'd62 && row==7'd62)begin
      next_row=7'd0;
      next_col=7'd0;
    end
    else if(col==7'd62)begin
      next_col=7'd0;
      next_row=row+7'd2;
    end
    else begin
      next_col=col+7'd2;
      next_row=row;
    end
  end
  else begin
    next_row=row;
    next_col=col;
  end
end

//store data
always@(posedge clk or posedge reset)begin
  if(reset)begin
    reg0<=20'd0;
    reg1<=20'd0;
    reg2<=20'd0;
    reg3<=20'd0;
    reg4<=20'd0;
    reg5<=20'd0;
    reg6<=20'd0;
    reg7<=20'd0;
    reg8<=20'd0;
  end
  else if(read9)begin
    case(read_counter)
        4'd0:reg0<=tmp_data;
        4'd1:reg1<=tmp_data;
        4'd2:reg2<=tmp_data;
        4'd3:reg3<=tmp_data;
        4'd4:reg4<=tmp_data;
        4'd5:reg5<=tmp_data;
        4'd6:reg6<=tmp_data;
        4'd7:reg7<=tmp_data;
        4'd8:reg8<=tmp_data;
        default:begin
        end
    endcase
  end
  else if(readc)begin
    case(read_counter)
        4'd0:begin
            reg0<=reg1;
            reg1<=reg2;
            reg2<=tmp_data;
            reg3<=reg4;
            reg4<=reg5;
            reg5<=20'd0;
            reg6<=reg7;
            reg7<=reg8;
            reg8<=20'd0;
        end
        4'd1:reg5<=tmp_data;
        4'd2:reg8<=tmp_data;
        default:begin
        end
    endcase
  end
  else begin
  end
end

// caculate address
always@(*)begin
  if(read9)begin
    case(read_counter)
      4'd0:begin
        tmp_row=row;
        tmp_col=col;
      end
      4'd1:begin
        tmp_row=row;
        tmp_col=col+7'd1;
      end
      4'd2:begin
        tmp_row=row;
        tmp_col=col+7'd2;
      end
      4'd3:begin
        tmp_row=row+7'd1;
        tmp_col=col;
      end
      4'd4:begin
        tmp_row=row+7'd1;
        tmp_col=col+7'd1;
      end
      4'd5:begin
        tmp_row=row+7'd1;
        tmp_col=col+7'd2;
      end
      4'd6:begin
        tmp_row=row+7'd2;
        tmp_col=col;
      end
      4'd7:begin
        tmp_row=row+7'd2;
        tmp_col=col+7'd1;
      end
      4'd8:begin
        tmp_row=row+7'd2;
        tmp_col=col+7'd2;
      end
      default:begin
        tmp_row=row;
        tmp_col=col;
      end
    endcase
  end
  else if(readc)begin
    case(read_counter)
        4'd0:begin
          tmp_row=row;
          tmp_col=col+7'd2;
        end
        4'd1:begin
          tmp_row=row+7'd1;
          tmp_col=col+7'd2;
        end
        4'd2:begin
          tmp_row=row+7'd2;
          tmp_col=col+7'd2;
        end
        default:begin
          tmp_row=row;
          tmp_col=col;
        end
    endcase
  end
  else begin
    tmp_row=row;
    tmp_col=col;
  end
end

//CONV layer
always@(posedge clk or posedge reset)begin
  if(reset)begin
    conv_count<=4'd0;
  end
  else if(conv_en)begin
    conv_count<=next_conv_count;
  end
  else begin
    conv_count<=4'd0;
  end
end
always@(*)begin
  if(conv_en)begin
    next_conv_count=conv_count+4'd1;
  end
  else begin
    next_conv_count=conv_count;
  end
end

always@(*)begin
  if(conv_en)begin
    case(conv_count)
        4'd0:begin
          img_data=reg0;
          weight=20'h0A89E;
        end
        4'd1:begin
          img_data=reg1;
          weight=20'h092D5;
        end
        4'd2:begin
          img_data=reg2;
          weight=20'h06D43;
        end
        4'd3:begin
          img_data=reg3;
          weight=20'h01004;
        end
        4'd4:begin
          img_data=reg4;
          weight=20'hF8F71;
        end
        4'd5:begin
          img_data=reg5;
          weight=20'hF6E54;
        end
        4'd6:begin
          img_data=reg6;
          weight=20'hFA6D7;
        end
        4'd7:begin
          img_data=reg7;
          weight=20'hFC834;
        end
        4'd8:begin
          img_data=reg8;
          weight=20'hFAC19;
        end
        default:begin
          img_data=20'd0;
          weight=20'd0;
        end
    endcase
  end
  else begin
    img_data=20'd0;
    weight=20'd0;
  end
end
always@(posedge clk or posedge reset)begin
  if(reset)begin
    psum<=40'd0;
  end
  else if(conv_en)begin
    psum<=next_psum;
  end
  else begin
    psum<=40'd0;
  end
end
always@(*)begin
  if(conv_en)begin
    next_psum=psum+tmp_result;
  end
  else begin
    next_psum=psum;
  end
end

//ReLU layer
always@(posedge clk or posedge reset)begin
  if(reset)begin
    conv_data<=20'd0;
  end
  else if(relu)begin
    conv_data<=next_conv_data;
  end
  else begin
    conv_data<=20'd0;
  end
end
always@(*)begin
  if(relu)begin
    addBias_result=psum+{4'd0,20'h01310,16'd0};
    if(addBias_result[39]==1'b1||addBias_result==40'd0)begin
      next_conv_data=20'd0;
    end
    else begin
      if(addBias_result[15]==1'b1)begin
        next_conv_data=addBias_result[35:16]+20'd1;
      end
      else begin
        next_conv_data=addBias_result[35:16];
      end
    end
  end
  else begin
    next_conv_data=conv_data;
  end
end

//max pooling layer
always@(posedge clk or posedge reset)begin
  if(reset)begin
    max<=20'd0;
  end
  else if(max_en)begin
    max<=n_max;
  end
  else begin
    max<=20'd0;
  end
end
always@(*)begin
  if(max_en)begin
    if(max<cdata_rd)begin
      n_max=cdata_rd;   
    end
    else begin
      n_max=max;
    end
    case(read_counter)
      4'd0:begin
        caddr_rd=max_addr;
      end
      4'd1:begin
        caddr_rd=max_addr+12'd1;
      end
      4'd2:begin
        caddr_rd=max_addr+12'd64;
      end
      4'd3:begin
        caddr_rd=max_addr+12'd65;
      end
      default:begin
        caddr_rd=max_addr;
      end
    endcase
  end
  else begin
    n_max=max;
    caddr_rd=max_addr;
  end
end

// write in memory
always@(posedge clk or posedge reset)begin
  if(reset)begin
    caddr_count<=12'd0;
  end
  else if(layer1_wen)begin
    caddr_count<=next_caddr_count;
  end
  else begin
  end
end
always@(*)begin
  if(layer1_wen)begin
    next_caddr_count=caddr_count+12'd1;
  end
  else begin
    next_caddr_count=caddr_count;
  end
end
always@(*)begin
  if(state==LAYER0_WRITE)begin
    caddr_wr=(row<<6)+col;
    cdata_wr=conv_data;
  end
  else if(state==LAYER1_WRITE)begin
    caddr_wr=caddr_count;
    cdata_wr=max;
  end
  else begin
    caddr_wr=12'd0;
    cdata_wr=20'd0;
  end
end

//memory choose
always@(*)begin
  case(state)
    LAYER0_WRITE:csel=3'b001;
    MAX_POOLING:csel=3'b001;
    LAYER1_WRITE:csel=3'b011;
    default:csel=3'b000;
  endcase
end

// busy signal
always@(posedge clk or posedge reset)begin
  if(reset)begin
    busy<=1'b0;
  end
  else if(ready)begin
    busy<=1'b1;
  end
  else if(state==LAYER1_DONE)begin
    busy<=1'b0;
  end
  else begin
  end
end


endmodule




