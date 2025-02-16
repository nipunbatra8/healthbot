import { z } from "zod";
import {
  defineDAINService,
  ToolConfig
} from "@dainprotocol/service-sdk";
import { AlertUIBuilder, CardUIBuilder, DainResponse } from "@dainprotocol/utils";

const askForHelp = {
  description: "Ask for help when critical health readings are detected",
  parameters: z.object({
    question: z.string().describe("The alert that needs attention")
  }),
  execute: async ({ question }, processId, processHandler, callSummary) => {
    // Log the help request
    callSummary.steps.push({
      speaker: "ask_for_help",
      text: question,
      timestamp: new Date().toISOString()
    });

    try {
      // Request human intervention
      const stepId = await processHandler.requestHumanAction(processId, {
        message: question,
        ui: new CardUIBuilder()
          .title("Critical Health Alert")
          .content(`I need help with: ${question}`)
          .build(),
        actions: [
          {
            id: "check-soldier",
            title: "Soldier checked - Status OK",
            requiresResponse: true
          },
          {
            id: "medical-needed",
            title: "Medical attention required",
            requiresResponse: true
          }
        ],
        timeoutMs: 30 * 1000
      });

      // Wait for response
      const response = await processHandler.waitForHumanAction(
        processId,
        stepId,
        30 * 1000
      );

      // Handle responses
      if (response.actionId === "medical-needed") {
        callSummary.steps.push({
          speaker: "supervisor",
          text: `Medical attention required: ${response.responseText}`,
          timestamp: new Date().toISOString()
        });
        return "Medical attention has been requested: " + response.responseText;
      }

      if (response.actionId === "check-soldier") {
        callSummary.steps.push({
          speaker: "supervisor",
          text: response.responseText || "Soldier status confirmed OK",
          timestamp: new Date().toISOString()
        });
        return "Soldier has been checked: " + response.responseText;
      }

    } catch (error) {
      return "No response received from supervisor. Please escalate if condition persists.";
    }
  }
};

// Sample PPG data generator with varied readings
function generatePPGData() {
  const isCritical = Math.random() < 0.3;
  
  let baseGreen, baseRed;
  
  if (isCritical) {
    baseGreen = 84000 + Math.random() * 1000;
    baseRed = 196000 + Math.random() * 1000;
  } else {
    baseGreen = 83000 + Math.random() * 500;
    baseRed = 195000 + Math.random() * 500;
  }
  
  return {
    green: baseGreen,
    red: baseRed,
    IR: 206000 + Math.random() * 1000,
    acc_x: 8 + Math.random(),
    acc_y: -4 + Math.random(),
    acc_z: 3 + Math.random(),
    timestamp: Date.now()
  };
}

const startMonitoringConfig: ToolConfig = {
  id: "start-monitoring",
  name: "Start Health Monitoring",
  description: "Begins real-time health monitoring using PPG data",
  input: z
    .object({
      teamMemberId: z.string().describe("ID of the team member to monitor"),
    })
    .describe("Monitoring parameters"),
  output: z
    .object({
      processId: z.string().describe("ID of the monitoring process"),
      status: z.string().describe("Status of monitoring"),
    })
    .describe("Monitoring information"),
  pricing: { pricePerUse: 0, currency: "USD" },
  handler: async ({ teamMemberId }, agentInfo, { app }) => {
    console.log(`Starting monitoring for team member ${teamMemberId}`);

    const processId = await app.processes!.createProcess(
      agentInfo,
      "recurring",
      "Health Monitoring",
      `Monitoring team member ${teamMemberId}`
    );

    const callSummary = {
      steps: []
    };

    // Start monitoring in background
    (async () => {
      try {
        let updateCount = 0;
        
        const interval = setInterval(async () => {
          const ppgData = generatePPGData();
          
          if (ppgData.green > 84000 || ppgData.red > 196000) {
            // Critical readings detected - request human intervention
            const question = `CRITICAL: High PPG levels detected for ${teamMemberId}\nGreen: ${Math.round(ppgData.green)}\nRed: ${Math.round(ppgData.red)}`;
            const response = await askForHelp.execute(
              { question }, 
              processId, 
              app.processes!, 
              callSummary
            );

            // Add critical update with human response
            await app.processes!.addUpdate(processId, {
              percentage: Math.min((updateCount / 100) * 100, 99),
              text: response,
              // data: {
              //   status: "critical",
              //   readings: ppgData,
              //   response: response
              // }
            });
          } else {
            // Normal update
            await app.processes!.addUpdate(processId, {
              percentage: Math.min((updateCount / 100) * 100, 99),
              text: `Normal readings - Green: ${Math.round(ppgData.green)}, Red: ${Math.round(ppgData.red)}`,
              // data: {
              //   status: "normal",
              //   readings: ppgData
              // }
            });
          }

          updateCount++;
          
          if (updateCount >= 20) {
            clearInterval(interval);
            await app.processes!.addResult(processId, {
              text: "Monitoring session completed",
              data: { 
                totalUpdates: updateCount,
                status: "completed",
                summary: callSummary
              }
            });
          }
        }, 3000);

      } catch (error) {
        await app.processes!.failProcess(processId, error.message);
      }
    })();

    const startAlertUI = new AlertUIBuilder()
      .variant('info')
      .title('Health Monitoring Started')
      .message(`Beginning health monitoring for team member ${teamMemberId}`)
      .icon(true)
      .build();

    return new DainResponse({
      text: "Health monitoring started",
      data: { 
        processId,
        status: "monitoring"
      },
      ui: startAlertUI,
      processes: [processId]
    });
  },
};

const dainService = defineDAINService({
  metadata: {
    title: "Health Monitoring Service",
    description: "Real-time health monitoring service using PPG data",
    version: "1.0.0",
    author: "Health Monitor",
    tags: ["health", "monitoring", "ppg"],
    logo: "https://cdn-icons-png.flaticon.com/512/2966/2966327.png",
  },
  exampleQueries: [
    {
      category: "Monitoring",
      queries: [
        'Start monitoring team member TM001',
        'Begin health tracking for soldier A123',
        'Monitor vitals for member B456',
      ],
    }
  ],
  identity: {
    apiKey: process.env.DAIN_API_KEY,
  },
  tools: [startMonitoringConfig],
});

dainService.startNode({ port: 2022 }).then(() => {
  console.log("Health Monitoring Service is running on port 2022");
});