// Carbon Imports
import {
  Breadcrumb,
  BreadcrumbItem,
  Tabs,
  Tab,
  TabList,
  TabPanels,
  TabPanel,
  Grid,
  Column,
} from '@carbon/react';
import ChatWindow from '../ChatWindow';

// Component Imports
import UploadFileComponent from '../UploadFileComponent';

const APPLICATION_NAME =
  process.env.APPLICATION_NAME || 'AI Applied Demo';

export default function LandingPage() {
  return (
    <Grid className='landing-page' fullWidth>
      <Column lg={16} md={8} sm={4} className='landing-page__banner'>
        <Breadcrumb aria-label='Page navigation'>
          <BreadcrumbItem className='landing-page__breadcrumb'>
            <a href='/'>Home</a>
          </BreadcrumbItem>
        </Breadcrumb>
        <h1 className='landing-page__heading'>{APPLICATION_NAME}</h1>
      </Column>
      <Column lg={16} md={8} sm={4} className='landing-page__r2'>
        <Tabs defaultSelectedIndex={0}>
          <TabList className='tabs-group' aria-label='Page navigation'>
            <Tab>Chat with your docs</Tab>
            <Tab>Upload Docs to Vector DB</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
              <Grid className='tabs-group-content' fullWidth>
                <Column sm={4} md={8} lg={16}>
                  <ChatWindow/>
                </Column>
              </Grid>
            </TabPanel>
            <TabPanel>
              <Grid className='tabs-group-content' fullWidth>
                <Column sm={4} md={8} lg={16}>
                  <UploadFileComponent/>
                </Column>
              </Grid>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Column>
    </Grid>
  );
}
