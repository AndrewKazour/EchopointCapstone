/*

 * STM32 motor executor for Raspberry Pi commands

 * Command format from Pi:

 *   MOVE <forward_pct> <turn_pct>\n

 * Example:

 *   MOVE 30 0     -> drive forward

 *   MOVE 0 20     -> slow right search turn

 *   MOVE 20 -15   -> forward while curving left

 *

 * forward_pct : -100 to +100

 * turn_pct    : -100 to +100   (positive = turn right)

 */



#include "main.h"

#include <string.h>

#include <stdlib.h>

#include <stdio.h>

#include <stdint.h>



TIM_HandleTypeDef htim3;

UART_HandleTypeDef huart1;



#define UART_BUF_SIZE      40

#define PWM_MAX            2560

#define CMD_TIMEOUT_MS     300

#define CONTROL_PERIOD_MS  10

#define RAMP_STEP_PCT      8

#define MIN_MOTOR_PCT     60



static uint8_t uart_byte;

static char rx_buf[UART_BUF_SIZE];

static uint8_t rx_idx = 0;

static volatile uint8_t line_ready = 0;

static char line_buf[UART_BUF_SIZE];



static int target_left_pct  = 0;

static int target_right_pct = 0;

static int actual_left_pct  = 0;

static int actual_right_pct = 0;

static uint32_t last_cmd_ms = 0;



void SystemClock_Config(void);

static void MX_GPIO_Init(void);

static void MX_TIM3_Init(void);

static void MX_USART1_UART_Init(void);



static int clamp_i(int value, int low, int high)

{

    if (value < low) return low;

    if (value > high) return high;

    return value;

}



static int ramp_toward(int current, int target, int step)

{

    if (current < target)

    {

        current += step;

        if (current > target) current = target;

    }

    else if (current > target)

    {

        current -= step;

        if (current < target) current = target;

    }

    return current;

}



static int enforce_min_motor_pct(int value)

{

    if (value == 0) return 0;



    int sign = (value > 0) ? 1 : -1;

    int mag = abs(value);



    if (mag < MIN_MOTOR_PCT)

        mag = MIN_MOTOR_PCT;



    return sign * mag;

}



static void set_motor_signed(int left_pct, int right_pct)

{

    left_pct  = clamp_i(left_pct,  -100, 100);

    right_pct = clamp_i(right_pct, -100, 100);



    uint8_t left_fwd  = (left_pct >= 0) ? 1 : 0;

    uint8_t right_fwd = (right_pct >= 0) ? 1 : 0;



    uint32_t left_pulse  = (uint32_t)(PWM_MAX * abs(left_pct)  / 100);

    uint32_t right_pulse = (uint32_t)(PWM_MAX * abs(right_pct) / 100);



    /* Right motor is mounted mirrored */

    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, left_fwd  ? GPIO_PIN_SET   : GPIO_PIN_RESET);

    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, right_fwd ? GPIO_PIN_RESET : GPIO_PIN_SET);



    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, left_pulse);

    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, right_pulse);

}



static void set_target_motion(int forward_pct, int turn_pct)

{

    int left  = forward_pct + turn_pct;

    int right = forward_pct - turn_pct;



    left  = clamp_i(left,  -100, 100);

    right = clamp_i(right, -100, 100);



    target_left_pct  = enforce_min_motor_pct(left);

    target_right_pct = enforce_min_motor_pct(right);

}



static void stop_targets(void)

{

    target_left_pct = 0;

    target_right_pct = 0;

}



static void parse_command(const char *line)

{

    int forward_pct = 0;

    int turn_pct = 0;



    if (strncmp(line, "STOP", 4) == 0)

    {

        stop_targets();

        last_cmd_ms = HAL_GetTick();

        return;

    }



    if (sscanf(line, "MOVE %d %d", &forward_pct, &turn_pct) == 2)

    {

        forward_pct = clamp_i(forward_pct, -100, 100);

        turn_pct    = clamp_i(turn_pct, -100, 100);

        set_target_motion(forward_pct, turn_pct);

        last_cmd_ms = HAL_GetTick();

        return;

    }



    {

        const char *msg = "ERR\r\n";

        HAL_UART_Transmit(&huart1, (uint8_t *)msg, strlen(msg), 50);

    }

}



void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)

{

    if (huart->Instance != USART1) return;



    char c = (char)uart_byte;



    if (c == '\n' || c == '\r')

    {

        if (rx_idx > 0 && !line_ready)

        {

            rx_buf[rx_idx] = '\0';

            memcpy(line_buf, rx_buf, rx_idx + 1);

            line_ready = 1;

            rx_idx = 0;

        }

    }

    else if (rx_idx < UART_BUF_SIZE - 1)

    {

        rx_buf[rx_idx++] = c;

    }

    else

    {

        rx_idx = 0;

    }



    HAL_UART_Receive_IT(&huart1, &uart_byte, 1);

}



int main(void)

{

    uint32_t last_control_ms = 0;



    HAL_Init();

    SystemClock_Config();

    MX_GPIO_Init();

    MX_TIM3_Init();

    MX_USART1_UART_Init();



    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);

    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);



    set_motor_signed(0, 0);

    HAL_UART_Receive_IT(&huart1, &uart_byte, 1);

    last_cmd_ms = HAL_GetTick();



    {

        const char *hello = "STM32 Ready\r\n";

        HAL_UART_Transmit(&huart1, (uint8_t *)hello, strlen(hello), 100);

    }



    while (1)

    {

        uint32_t now = HAL_GetTick();



        if (line_ready)

        {

            line_ready = 0;

            parse_command(line_buf);

        }



        if ((now - last_cmd_ms) > CMD_TIMEOUT_MS)

        {

            stop_targets();

        }



        if ((now - last_control_ms) >= CONTROL_PERIOD_MS)

        {

            last_control_ms = now;



            actual_left_pct  = ramp_toward(actual_left_pct,  target_left_pct,  RAMP_STEP_PCT);

            actual_right_pct = ramp_toward(actual_right_pct, target_right_pct, RAMP_STEP_PCT);

            set_motor_signed(actual_left_pct, actual_right_pct);

        }

    }

}



void SystemClock_Config(void)

{

    RCC_OscInitTypeDef RCC_OscInitStruct = {0};

    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};



    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;

    RCC_OscInitStruct.HSEState       = RCC_HSE_ON;

    RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;

    RCC_OscInitStruct.HSIState       = RCC_HSI_ON;

    RCC_OscInitStruct.PLL.PLLState   = RCC_PLL_ON;

    RCC_OscInitStruct.PLL.PLLSource  = RCC_PLLSOURCE_HSE;

    RCC_OscInitStruct.PLL.PLLMUL     = RCC_PLL_MUL4;

    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)

        Error_Handler();



    RCC_ClkInitStruct.ClockType      = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK

                                     | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;

    RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;

    RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;

    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;

    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)

        Error_Handler();



    HAL_RCC_EnableCSS();

}



static void MX_TIM3_Init(void)

{

    TIM_MasterConfigTypeDef sMasterConfig = {0};

    TIM_OC_InitTypeDef sConfigOC = {0};



    htim3.Instance = TIM3;

    htim3.Init.Prescaler = 0;

    htim3.Init.CounterMode = TIM_COUNTERMODE_UP;

    htim3.Init.Period = 2559;

    htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;

    htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;

    if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)

        Error_Handler();



    sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;

    sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;

    if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)

        Error_Handler();



    sConfigOC.OCMode = TIM_OCMODE_PWM2;

    sConfigOC.Pulse = 0;

    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;

    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;

    if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)

        Error_Handler();

    if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)

        Error_Handler();



    HAL_TIM_MspPostInit(&htim3);

}



static void MX_USART1_UART_Init(void)

{

    huart1.Instance = USART1;

    huart1.Init.BaudRate = 115200;

    huart1.Init.WordLength = UART_WORDLENGTH_8B;

    huart1.Init.StopBits = UART_STOPBITS_1;

    huart1.Init.Parity = UART_PARITY_NONE;

    huart1.Init.Mode = UART_MODE_TX_RX;

    huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;

    huart1.Init.OverSampling = UART_OVERSAMPLING_16;

    if (HAL_UART_Init(&huart1) != HAL_OK)

        Error_Handler();



    HAL_NVIC_SetPriority(USART1_IRQn, 0, 0);

    HAL_NVIC_EnableIRQ(USART1_IRQn);

}



static void MX_GPIO_Init(void)

{