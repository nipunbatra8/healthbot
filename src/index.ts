import { number, z } from "zod";
import {
  defineDAINService,
  ToolConfig
} from "@dainprotocol/service-sdk";
import { AlertUIBuilder, CardUIBuilder, DainResponse } from "@dainprotocol/utils";
import test from "node:test";

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



const Papa = require("papaparse");
const fs = require("fs");

let csvData = [];
let currentIndex = 1; // Start from 1 to skip the header row

// Load the CSV file once
function loadCSV() {
  const csvFile = fs.readFileSync(
    "/Users/adarshashok/Downloads/treehacks/healthbot/src/test_1.csv",
    "utf8"
  );

  // Parse the CSV data
  const results = Papa.parse(csvFile, {
    header: false,
    dynamicTyping: true,
  });

  csvData = results.data;
}

// Call loadCSV() once when the script starts
loadCSV();

async function createPPGData(number_of_soldiers) {
  const createdFiles = [];
  const regularFilePath = '/Users/adarshashok/Downloads/treehacks/healthbot/src/regular.csv';
  const tempDir = path.join(process.cwd(), 'temp');

  // Ensure the 'temp' directory exists
  fs.mkdirSync(tempDir, { recursive: true });

  try {
    // Read the first 501 lines from regular.csv
    const fileStream = fs.createReadStream(regularFilePath);
    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });

    const lines = [];
    for await (const line of rl) {
      if (lines.length < 501) {
        lines.push(line);
      } else {
        break;
      }
    }

    if (lines.length === 0) {
      throw new Error('regular.csv is empty or does not have 501 lines.');
    }

    // Ensure the last line ends with a newline
    const fileContent = lines.join('\n') + '\n';

    // Create the soldier files with the first 501 lines from regular.csv
    for (let i = 0; i < number_of_soldiers; i++) {
      const fileName = `file_${i + 1}.csv`;
      const filePath = path.join(tempDir, fileName);

      // Write the collected lines to the new file
      fs.writeFileSync(filePath, fileContent, 'utf8');

      createdFiles.push(fileName);
      console.log(`Created file: ${fileName} with 501 lines.`);
    }
  } catch (error) {
    console.error(`Error: ${error.message}`);
  }

  return createdFiles;
}

let lastCopiedLine = 500; 
async function addData(number_of_soldiers: number) {

  for (let i = 0; i < number_of_soldiers; i++) {
    const fileName = `/Users/adarshashok/Downloads/treehacks/healthbot/temp/file_${i + 1}.csv`;
    const testName = `/Users/adarshashok/Downloads/treehacks/healthbot/src/test_${i + 1}.csv`
    await copyDataPoints(testName, fileName, lastCopiedLine);
  }
  lastCopiedLine += 10;

}




const { spawn } = require("child_process");
const path = require("path");

/**
 * Runs a Python script on multiple CSV files and logs the output.
 * @param {string[]} csvFiles - Array of CSV file paths.
 */
function processCSVFiles(csvFiles) {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, "heart_predictor.py"); // Path to Python script

    const pythonProcess = spawn("python3", [pythonScript, ...csvFiles]);

    let outputData = '';

    // Handle stdout (Python output)
    pythonProcess.stdout.on("data", (data) => {
      console.log(`Python Output: ${data.toString().trim()}`);
      outputData += data.toString();
    });

    // Handle stderr (errors)
    pythonProcess.stderr.on("data", (data) => {
      console.error(`Python Error: ${data.toString().trim()}`);
    });

    // Handle process exit
    pythonProcess.on("close", (code) => {
      console.log(`Python process exited with code ${code}`);
      if (code === 0) {
        // Split output into an array, assuming Python output is comma-separated
        resolve(outputData.trim().split(/\s*,\s*/));
      } else {
        reject(new Error(`Python script exited with code ${code}`));
      }
    });
  });
}


const readline = require('readline');

/**
 * Copies 10 data points from the source CSV and appends to the target CSV,
 * incrementing the start position each time.
 * @param {string} sourceFile - Path to the source CSV.
 * @param {string} targetFile - Path to the target CSV.
 */
async function copyDataPoints(sourceFile, targetFile, lastCopiedLine) {
  try {
    if (!fs.existsSync(sourceFile)) {
      throw new Error(`Source file not found: ${sourceFile}`);
    }
    if (!fs.existsSync(targetFile)) {
      console.warn(`Target file not found. Creating: ${targetFile}`);
      fs.writeFileSync(targetFile, '');
    }

    // Read the source CSV file
    const inputStream = fs.createReadStream(sourceFile);
    const rl = readline.createInterface({
      input: inputStream,
      crlfDelay: Infinity,
    });

    let rowsToCopy = [];
    let currentLine = 0;

    for await (const line of rl) {
      if (currentLine >= lastCopiedLine && currentLine < lastCopiedLine + 10) {
        rowsToCopy.push(line);
      }
      currentLine++;

      if (rowsToCopy.length === 10) break;
    }

    if (rowsToCopy.length === 0) {
      console.log('No more data to copy.');
      return;
    }

    // Append data to target CSV
    fs.appendFileSync(targetFile, rowsToCopy.join('\n') + '\n');
    console.log(`Copied ${rowsToCopy.length} rows from line ${lastCopiedLine} to ${targetFile}`);

    lastCopiedLine += 10; // Update the position for the next run

  } catch (error) {
    console.error(`Error: ${error.message}`);
  }
}

function generatePPGData() {
  const isCritical = Math.random() < 0.15;
  
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
      teamMemberId_1: z.string().describe("ID of the first team member to monitor"),
      teamMemberId_2: z.string().optional().describe("ID of the second member to monitor"),
      teamMemberId_3: z.string().optional().describe("ID of the third team member to monitor"),
      teamMemberId_4: z.string().optional().describe("ID of the fourth team member to monitor"),
      teamMemberId_5: z.string().optional().describe("ID of the fifth team member to monitor"),
    })
    .describe("Monitoring parameters"),
  output: z
    .object({
      processId: z.string().describe("ID of the monitoring process"),
      status: z.string().describe("Status of monitoring"),
    })
    .describe("Monitoring information"),
  pricing: { pricePerUse: 0, currency: "USD" },
  handler: async ({ teamMemberId_1, teamMemberId_2, teamMemberId_3, teamMemberId_4, teamMemberId_5 }, agentInfo, { app }) => {
    const teamMemberIds = [teamMemberId_1, teamMemberId_2, teamMemberId_3, teamMemberId_4, teamMemberId_5].filter(Boolean);
    const processId = await app.processes!.createProcess(
      agentInfo,
      "one-time",
      "Health Monitoring",
      `Monitoring ${teamMemberIds.length} team members`
    );
    const callSummary = {
      steps: []
    };
    await createPPGData(teamMemberIds.length);

    (async () => {
      try {
        let updateCount = 0;
        
        const interval = setInterval(async () => {
          await addData(teamMemberIds.length);
          const items: string[] = [];
          for (let i = 0; i < teamMemberIds.length; i++) {
            items.push(`/Users/adarshashok/Downloads/treehacks/healthbot/temp/file_${i + 1}.csv`);
          }
          processCSVFiles(items)
            .then((result) => {
              console.log("Returned array:", result);
            })
            .catch((error) => {
              console.error("Error:", error);
            });
          const ppgData = generatePPGData();
          
          if (ppgData.green > 84000 || ppgData.red > 196000) {
            const randomIndex = Math.floor(Math.random() * teamMemberIds.length);
            
  
            // Critical readings detected - request human intervention
            const question = `CRITICAL: High PPG levels detected for ${teamMemberIds[randomIndex]}\nGreen: ${Math.round(ppgData.green)}\nRed: ${Math.round(ppgData.red)}`;
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
        }, 4000);

        // const interval = setInterval(async () => {
        //   await addData(teamMemberIds.length);
        //   const items: string[] = [];
        //   for (let i = 0; i < teamMemberIds.length; i++) {
        //     items.push(`/Users/adarshashok/Downloads/treehacks/healthbot/temp/file_${i + 1}.csv`);
        //   }
        //   processCSVFiles(items)
        //     .then((result) => {
        //       console.log("Returned array:", result);
        //     })
        //     .catch((error) => {
        //       console.error("Error:", error);
        //     });
        //     if (Math.random() > 0.7) {
        //       const question = `CRITICAL: High PPG levels detected for ${teamMemberId_1}\nGreen: ${Math.round(ppgData.green)}\nRed: ${Math.round(ppgData.red)}`;
        //       const response = await askForHelp.execute(
        //         { question }, 
        //         processId, 
        //         app.processes!, 
        //         callSummary
        //       );

        //       // Add critical update with human response
        //       await app.processes!.addUpdate(processId, {
        //         percentage: Math.min((updateCount / 100) * 100, 99),
        //         text: response,
        //         // data: {
        //         //   status: "critical",
        //         //   readings: ppgData,
        //         //   response: response
        //         // }
        //       });
        //     } else {
        //       console.log("Random value is not greater than 0.7");
        //     }
            
        //   // count++;

        //   // await app.processes!.addUpdate(processId, {
        //   //   percentage: Math.min((count / maxCount) * 100, 99),
        //   //   text: `Task performed ${count} times`,
        //   // });

          
        // }, 6000); // 5 seconds interval

      } catch (error) {
        await app.processes!.failProcess(processId, error.message);
      }
    })();
    const startAlertUI = new AlertUIBuilder()
      .variant('info')
      .title('Health Monitoring Started')
      .message('Beginning health monitoring for team members')
      .icon(true)
      .build();


    return new DainResponse({
      text: `Started monitoring for ${teamMemberIds.length} team members`,
      data: {
        processId,
        status: "Monitoring started"
      },
      ui: new CardUIBuilder()
        .title("Health Monitoring Started")
        .content(`Monitoring process initiated for ${teamMemberIds.length} team members.`)
        .build()
    });
  }
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